#include "pvc/CudaSplitBregmanPVC.hpp"
#include "pvc/CudaOperators.cuh"
#include "pvc/DeviceVolume3D.cuh"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>

namespace pvc {
namespace {

// ---------------------------------------------------------------------------
// Thrust-based global reductions (safe, no naive atomics for full-volume sums)
// ---------------------------------------------------------------------------

struct SquareOp {
    __host__ __device__ double operator()(double x) const { return x * x; }
};

// L2 norm of a single device volume
double deviceL2Norm(const cuda::DeviceVolume3D &v) {
    thrust::device_ptr<const double> p(v.ptr);
    double sum_sq = thrust::transform_reduce(p, p + v.size(), SquareOp(), 0.0, thrust::plus<double>());
    return std::sqrt(sum_sq);
}

// L2 norm of a concatenated vector field (sqrt(sum of squares of all 3 components))
double deviceVectorFieldNorm(const cuda::DeviceVectorField3D &vf) {
    thrust::device_ptr<const double> px(vf.x.ptr), py(vf.y.ptr), pz(vf.z.ptr);
    double sx = thrust::transform_reduce(px, px + vf.x.size(), SquareOp(), 0.0, thrust::plus<double>());
    double sy = thrust::transform_reduce(py, py + vf.y.size(), SquareOp(), 0.0, thrust::plus<double>());
    double sz = thrust::transform_reduce(pz, pz + vf.z.size(), SquareOp(), 0.0, thrust::plus<double>());
    return std::sqrt(sx + sy + sz);
}

// Max element
double deviceMax(const cuda::DeviceVolume3D &v) {
    thrust::device_ptr<const double> p(v.ptr);
    return *thrust::max_element(p, p + v.size());
}

// ---------------------------------------------------------------------------
// Device-side objective + gradient (mirrors CPU objectiveAndGradient)
// ---------------------------------------------------------------------------
struct DeviceObjGradResult {
    double value;
    // gradient is written into caller-provided buffer
};

// All scratch buffers are pre-allocated and passed in to avoid per-iteration mallocs.
struct ScratchBuffers {
    // Convolution temporaries
    cuda::DeviceVolume3D conv_tmp1, conv_tmp2;
    // Gradient outputs (reused across calls)
    cuda::DeviceVectorField3D gu;      // gradient of u
    cuda::DeviceVectorField3D b_gu;    // PLS-projected gradient of u
    cuda::DeviceVectorField3D fx2;     // d - b_gu - bregman
    cuda::DeviceVectorField3D b_fx2;   // PLS-projected fx2
    cuda::DeviceVolume3D residual;     // blurred - pet
    cuda::DeviceVolume3D dfx1;         // conv(residual, psf)  (data-fidelity gradient part)
    cuda::DeviceVolume3D dfx2_vol;     // divergence(b_fx2)    (penalty gradient part)
    cuda::DeviceVolume3D blurred;      // PSF-blurred u

    // Lambda update scratch
    cuda::DeviceVectorField3D r_prim;  // b_gu - d
    cuda::DeviceVectorField3D d_tmp;   // d_old - d
    cuda::DeviceVectorField3D b_dd;    // PLS(d_tmp)
    cuda::DeviceVolume3D dual_div;     // divergence(b_dd)
    cuda::DeviceVectorField3D b_gb;    // PLS(bregman)
    cuda::DeviceVolume3D factor_div;   // divergence(b_gb)

    // Line search scratch
    cuda::DeviceVolume3D dfx_old_grad; // gradient at u_old
    cuda::DeviceVolume3D test_step;    // trial u for backtracking

    // Scalar reduction buffer (device-side)
    double *d_penalty = nullptr;
    double *d_sdy = nullptr;
    double *d_ydy = nullptr;
    double *d_gsq = nullptr;

    ScratchBuffers(std::size_t nx, std::size_t ny, std::size_t nz)
        : conv_tmp1(nx, ny, nz), conv_tmp2(nx, ny, nz),
          gu(nx, ny, nz), b_gu(nx, ny, nz),
          fx2(nx, ny, nz), b_fx2(nx, ny, nz),
          residual(nx, ny, nz), dfx1(nx, ny, nz), dfx2_vol(nx, ny, nz),
          blurred(nx, ny, nz),
          r_prim(nx, ny, nz), d_tmp(nx, ny, nz), b_dd(nx, ny, nz),
          dual_div(nx, ny, nz), b_gb(nx, ny, nz), factor_div(nx, ny, nz),
          dfx_old_grad(nx, ny, nz), test_step(nx, ny, nz)
    {
        cudaMalloc(&d_penalty, sizeof(double));
        cudaMalloc(&d_sdy, sizeof(double));
        cudaMalloc(&d_ydy, sizeof(double));
        cudaMalloc(&d_gsq, sizeof(double));
    }

    ~ScratchBuffers() {
        cudaFree(d_penalty);
        cudaFree(d_sdy);
        cudaFree(d_ydy);
        cudaFree(d_gsq);
    }

    ScratchBuffers(const ScratchBuffers &) = delete;
    ScratchBuffers &operator=(const ScratchBuffers &) = delete;
};

// Compute objective value and gradient, writing gradient into `grad`.
DeviceObjGradResult objectiveAndGradient(
    const cuda::DeviceVolume3D &pet_scaled,
    const cuda::DeviceVolume3D &u,
    const cuda::DeviceVectorField3D &gv,
    const double *h_kx, int kx_len,
    const double *h_ky, int ky_len,
    const double *h_kz, int kz_len,
    const cuda::DeviceVectorField3D &d,
    const cuda::DeviceVectorField3D &bregman,
    double mu, double lambda,
    cuda::DeviceVolume3D &grad,
    ScratchBuffers &S)
{
    // blurred = conv(u, psf)
    cuda::separableConvolve3D(u, h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                              S.conv_tmp1, S.conv_tmp2, S.blurred);

    // residual = blurred - pet_scaled
    cuda::addScaled(S.blurred, pet_scaled, -1.0, S.residual);

    // gu = forward_grad(u),  b_gu = PLS(gv, gu)
    cuda::gradient3DForward(u, S.gu.x, S.gu.y, S.gu.z);
    cuda::applyPLSOperator(gv.x, gv.y, gv.z,
                           S.gu.x, S.gu.y, S.gu.z,
                           S.b_gu.x, S.b_gu.y, S.b_gu.z);

    // fx2 = d - b_gu - bregman,  penalty = sum(fx2²)
    cuda::computeFx2AndPenalty(d.x, d.y, d.z,
                               S.b_gu.x, S.b_gu.y, S.b_gu.z,
                               bregman.x, bregman.y, bregman.z,
                               S.fx2.x, S.fx2.y, S.fx2.z,
                               S.d_penalty);

    double h_penalty = 0.0;
    cudaMemcpy(&h_penalty, S.d_penalty, sizeof(double), cudaMemcpyDeviceToHost);

    double residual_norm = deviceL2Norm(S.residual);
    double data_term = residual_norm * residual_norm;
    double obj = (mu / 2.0) * data_term + (lambda / 2.0) * h_penalty;

    // b_fx2 = PLS(gv, fx2)
    cuda::applyPLSOperator(gv.x, gv.y, gv.z,
                           S.fx2.x, S.fx2.y, S.fx2.z,
                           S.b_fx2.x, S.b_fx2.y, S.b_fx2.z);

    // dfx1 = conv(residual, psf)   [gradient of data fidelity]
    cuda::separableConvolve3D(S.residual, h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                              S.conv_tmp1, S.conv_tmp2, S.dfx1);

    // dfx2 = div(b_fx2)
    cuda::sumDiagonalBackwardDivergence(S.b_fx2.x, S.b_fx2.y, S.b_fx2.z, S.dfx2_vol);

    // grad = mu * dfx1 + lambda * dfx2
    cuda::fusedGradientCombine(S.dfx1, S.dfx2_vol, mu, lambda, grad);

    return {obj};
}

struct LambdaUpdateResult {
    double lambda;
    double primal_residual_normalized;
    double dual_residual_normalized;
};

LambdaUpdateResult lambdaUpdate(
    const cuda::DeviceVectorField3D &d,
    const cuda::DeviceVectorField3D &d_old,
    double lambda,
    const cuda::DeviceVectorField3D &gv,
    const cuda::DeviceVectorField3D &b_gu,
    const cuda::DeviceVectorField3D &bregman,
    int flag, double lambda_init,
    ScratchBuffers &S)
{
    // r_prim = b_gu - d
    cuda::computePrimalResidual(b_gu.x, b_gu.y, b_gu.z,
                                d.x, d.y, d.z,
                                S.r_prim.x, S.r_prim.y, S.r_prim.z);

    double r_prim_norm = deviceVectorFieldNorm(S.r_prim);
    double factor1 = deviceVectorFieldNorm(b_gu);
    double factor2 = deviceVectorFieldNorm(d);
    double r_prim_normalized = (std::max(factor1, factor2) > 0.0)
        ? r_prim_norm / std::max(factor1, factor2) : 0.0;

    // d_tmp = d_old - d
    cuda::vectorFieldSubtract(d_old.x, d_old.y, d_old.z,
                              d.x, d.y, d.z,
                              S.d_tmp.x, S.d_tmp.y, S.d_tmp.z);

    // b_dd = PLS(gv, d_tmp)
    cuda::applyPLSOperator(gv.x, gv.y, gv.z,
                           S.d_tmp.x, S.d_tmp.y, S.d_tmp.z,
                           S.b_dd.x, S.b_dd.y, S.b_dd.z);

    // dual_div = divergence(b_dd) * lambda
    cuda::sumDiagonalBackwardDivergence(S.b_dd.x, S.b_dd.y, S.b_dd.z, S.dual_div);
    cuda::scaleInPlace(S.dual_div, lambda);
    double r_dual_norm = deviceL2Norm(S.dual_div);

    // b_gb = PLS(gv, bregman)
    cuda::applyPLSOperator(gv.x, gv.y, gv.z,
                           bregman.x, bregman.y, bregman.z,
                           S.b_gb.x, S.b_gb.y, S.b_gb.z);
    cuda::sumDiagonalBackwardDivergence(S.b_gb.x, S.b_gb.y, S.b_gb.z, S.factor_div);
    double factor_dual = deviceL2Norm(S.factor_div);

    double r_dual_normalized = (factor_dual > 0.0) ? r_dual_norm / factor_dual : 0.0;

    double ratio_raw = (r_dual_normalized > 0.0) ? r_prim_normalized / r_dual_normalized : 1.0;
    double alpha_tmp = std::sqrt(std::max(ratio_raw, std::numeric_limits<double>::epsilon()));
    constexpr double alpha_max = 100.0;
    double alpha = alpha_max;
    if (alpha_tmp >= 1.0 && alpha_tmp < alpha_max) {
        alpha = alpha_tmp;
    } else if (alpha_tmp > 1.0 / alpha_max && alpha_tmp < 1.0) {
        alpha = 1.0 / alpha_tmp;
    }

    constexpr double beta = 8.0;
    if (flag == 0) {
        if (r_prim_norm > beta * r_dual_norm) {
            lambda *= alpha;
        } else if (r_dual_norm > beta * r_prim_norm) {
            lambda /= alpha;
        }
    } else {
        lambda = lambda_init;
    }

    return {lambda, r_prim_normalized, r_dual_normalized};
}

// Line search (Barzilai-Borwein with non-monotone Armijo backtracking)
double lineSearch(
    const cuda::DeviceVolume3D &pet_scaled,
    const cuda::DeviceVolume3D &u_current,
    const cuda::DeviceVolume3D &u_old,
    const cuda::DeviceVectorField3D &gv,
    const double *h_kx, int kx_len,
    const double *h_ky, int ky_len,
    const double *h_kz, int kz_len,
    const cuda::DeviceVectorField3D &d,
    const cuda::DeviceVectorField3D &bregman,
    double mu, double lambda,
    double c_value,
    const cuda::DeviceVolume3D &gradient,
    const SolverOptions &options,
    ScratchBuffers &S)
{
    // Compute gradient at u_old
    objectiveAndGradient(pet_scaled, u_old, gv,
                         h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                         d, bregman, mu, lambda, S.dfx_old_grad, S);

    // Compute BB step via fused reduction
    cuda::lineSearchDots(u_current, u_old, gradient, S.dfx_old_grad,
                         S.d_sdy, S.d_ydy, S.d_gsq);

    double h_sdy = 0.0, h_ydy = 0.0, h_gsq = 0.0;
    cudaMemcpy(&h_sdy, S.d_sdy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ydy, S.d_ydy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gsq, S.d_gsq, sizeof(double), cudaMemcpyDeviceToHost);

    double alpha = (std::abs(h_ydy) > 0.0) ? h_sdy / h_ydy : 1.0;
    if (!(alpha > 0.0) || !std::isfinite(alpha)) alpha = 1.0;

    // Backtracking
    for (int i = 0; i < options.line_search_max_backtracks; ++i) {
        // test_step = u_current - alpha * gradient
        cuda::addScaled(u_current, gradient, -alpha, S.test_step);

        auto res = objectiveAndGradient(pet_scaled, S.test_step, gv,
                                        h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                                        d, bregman, mu, lambda, S.dfx_old_grad, S);
        if (res.value <= c_value - options.line_search_epsilon * alpha * h_gsq) {
            return alpha;
        }
        alpha *= options.line_search_rho;
    }
    return alpha;
}

} // anonymous namespace

// ===========================================================================
// Public entry point
// ===========================================================================

SolverResult runSplitBregmanPVC_CUDA(const Volume3D &pet,
                                     const Volume3D &guidance_image,
                                     const SeparableKernel3D &psf,
                                     const SolverOptions &options) {
    pet.requireSameShape(guidance_image, "runSplitBregmanPVC_CUDA");
    if (pet.empty()) throw std::runtime_error("Input PET volume is empty.");

    const double scale_factor_pet = pet.max();
    if (scale_factor_pet <= 0.0)
        throw std::runtime_error("PET max intensity must be positive.");

    const std::size_t nx = pet.nx(), ny = pet.ny(), nz = pet.nz();
    const std::size_t N = nx * ny * nz;

    // ---- Host-side scaling (cheap, done once) ----
    std::vector<double> pet_data = pet.data();
    for (double &v : pet_data) v /= scale_factor_pet;

    std::vector<double> guid_data = guidance_image.data();
    double guid_max = *std::max_element(guid_data.begin(), guid_data.end());
    if (guid_max > 0.0) for (double &v : guid_data) v /= guid_max;

    // ---- Allocate ALL device memory BEFORE the loop ----
    cuda::DeviceVolume3D d_pet(nx, ny, nz);
    d_pet.copyFromHost(pet_data);

    cuda::DeviceVolume3D d_u(nx, ny, nz);
    d_u.copyFromHost(pet_data);  // u starts as pet_scaled

    cuda::DeviceVolume3D d_u_old(nx, ny, nz);
    d_u_old.zero();

    cuda::DeviceVolume3D d_u_current(nx, ny, nz); // snapshot for each iter
    cuda::DeviceVolume3D d_grad(nx, ny, nz);       // current objective gradient
    cuda::DeviceVolume3D d_delta(nx, ny, nz);      // u - u_current (convergence check)

    cuda::DeviceVectorField3D d_d(nx, ny, nz);
    d_d.zero();
    cuda::DeviceVectorField3D d_d_old(nx, ny, nz);

    cuda::DeviceVectorField3D d_bregman(nx, ny, nz);
    d_bregman.zero();

    // Build normalised guidance gradient on device
    cuda::DeviceVectorField3D d_gv(nx, ny, nz);
    {
        cuda::DeviceVolume3D d_guid(nx, ny, nz);
        d_guid.copyFromHost(guid_data);
        cuda::gradient3DForward(d_guid, d_gv.x, d_gv.y, d_gv.z);
        cuda::normalizeGuidanceGradient(d_gv.x, d_gv.y, d_gv.z, options.mr_smooth);
        // d_guid freed here (RAII)
    }

    ScratchBuffers S(nx, ny, nz);

    // PSF kernel arrays (host pointers passed each call — copied to constant mem)
    const double *h_kx = psf.kx.data();
    const double *h_ky = psf.ky.data();
    const double *h_kz = psf.kz.data();
    const int kx_len = static_cast<int>(psf.kx.size());
    const int ky_len = static_cast<int>(psf.ky.size());
    const int kz_len = static_cast<int>(psf.kz.size());

    // ---- Initial objective ----
    auto init = objectiveAndGradient(d_pet, d_u, d_gv,
                                     h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                                     d_d, d_bregman, options.mu, options.lambda_init,
                                     d_grad, S);
    double c_value = init.value;
    double p_value = 1.0;
    double lambda = options.lambda_init;

    int stop_counter = 0;
    int flag = 0;
    const double epsilon = options.epsilon_scale * static_cast<double>(N) /
                           static_cast<double>(181 * 210 * 181);

    SolverSummary summary{};
    summary.final_lambda = lambda;

    // ==================================================================
    // MAIN LOOP — entirely on device, no host↔device volume transfers
    // ==================================================================
    for (int iter = 1; iter <= options.niter; ++iter) {
        // Snapshot u_current = u
        d_u_current.copyFrom(d_u);

        // Objective + gradient at u_current
        auto current = objectiveAndGradient(d_pet, d_u_current, d_gv,
                                            h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                                            d_d, d_bregman, options.mu, lambda,
                                            d_grad, S);

        // Line search
        double alpha = lineSearch(d_pet, d_u_current, d_u_old, d_gv,
                                  h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                                  d_d, d_bregman, options.mu, lambda,
                                  c_value, d_grad, options, S);

        // u = u_current - alpha * grad
        cuda::addScaled(d_u_current, d_grad, -alpha, d_u);

        // Forward gradient + PLS projection of u
        cuda::gradient3DForward(d_u, S.b_gu.x, S.b_gu.y, S.b_gu.z);
        cuda::applyPLSOperator(d_gv.x, d_gv.y, d_gv.z,
                               S.b_gu.x, S.b_gu.y, S.b_gu.z,
                               S.gu.x, S.gu.y, S.gu.z);
        // gu now holds b_gu result; swap names for clarity
        // (S.gu is used as the "projected gradient" here)

        // Save d_old
        d_d_old.copyFrom(d_d);

        // Fused shrink + Bregman update (reads b_gu from S.gu, updates d and bregman in-place)
        cuda::fusedShrinkBregman(S.gu.x, S.gu.y, S.gu.z,
                                 d_d.x, d_d.y, d_d.z,
                                 d_bregman.x, d_bregman.y, d_bregman.z,
                                 lambda);

        // Lambda update (needs b_gu = S.gu)
        auto lambda_info = lambdaUpdate(d_d, d_d_old, lambda, d_gv, S.gu, d_bregman,
                                        flag, options.lambda_init, S);
        lambda = lambda_info.lambda;

        // Objective at new u for non-monotone step acceptance
        auto fx_res = objectiveAndGradient(d_pet, d_u, d_gv,
                                           h_kx, kx_len, h_ky, ky_len, h_kz, kz_len,
                                           d_d, d_bregman, options.mu, lambda,
                                           d_grad, S);
        double fx = fx_res.value;
        double p_new = options.eta * p_value + 1.0;
        c_value = (options.eta * p_value * c_value + fx) / p_new;
        p_value = p_new;

        // u_old = u_current
        d_u_old.copyFrom(d_u_current);

        // Convergence check: ||u - u_current||₂ / ||u_current||₂
        cuda::addScaled(d_u, d_u_current, -1.0, d_delta);
        double delta_norm = deviceL2Norm(d_delta);
        double u_current_norm = deviceL2Norm(d_u_current);
        double rel_update = (u_current_norm > 0.0) ? delta_norm / u_current_norm : 0.0;

        if (options.verbose) {
            std::cout << "finish iteration " << iter
                      << ": ||x(k)-x(k-1)||_2 / ||x(k-1)||_2 = "
                      << std::fixed << std::setprecision(3) << rel_update * 100.0 << "%"
                      << ", lambda = " << lambda
                      << ", rp = " << lambda_info.primal_residual_normalized
                      << ", rd = " << lambda_info.dual_residual_normalized
                      << '\n';
        }

        if (rel_update < options.convergence_tol) {
            ++stop_counter;
        } else {
            stop_counter = 0;
        }

        summary.iterations_run = iter;
        summary.final_lambda = lambda;
        summary.final_primal_residual = lambda_info.primal_residual_normalized;
        summary.final_dual_residual = lambda_info.dual_residual_normalized;

        if (stop_counter >= options.stop_counter_limit && flag <= 1) {
            ++flag;
            stop_counter = 0;
        } else if (lambda_info.primal_residual_normalized < epsilon && flag > 1) {
            break;
        }
    }

    // ==================================================================
    // SINGLE device→host transfer of the final corrected volume
    // ==================================================================
    cuda::scaleInPlace(d_u, scale_factor_pet);

    Volume3D out(nx, ny, nz, 0.0);
    d_u.copyToHost(out.data());

    return {out, summary};
}

} // namespace pvc
