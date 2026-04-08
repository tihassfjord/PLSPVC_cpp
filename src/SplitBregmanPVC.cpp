#include "pvc/SplitBregmanPVC.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "pvc/Operators.hpp"

namespace pvc {
namespace {

struct LambdaUpdateResult {
    double lambda = 0.0;
    double primal_residual_normalized = 0.0;
    double dual_residual_normalized = 0.0;
};

VectorField3D normalizeGuidanceGradient(const Volume3D &guidance, double mr_smooth) {
    VectorField3D gv = gradient3D(guidance, "forward");
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(guidance.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(guidance.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(guidance.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                const double gx = gv.x(xi, yi, zi);
                const double gy = gv.y(xi, yi, zi);
                const double gz = gv.z(xi, yi, zi);
                const double norm = std::sqrt(gx * gx + gy * gy + gz * gz + mr_smooth * mr_smooth);
                gv.x(xi, yi, zi) = gx / norm;
                gv.y(xi, yi, zi) = gy / norm;
                gv.z(xi, yi, zi) = gz / norm;
            }
        }
    }
    return gv;
}

double vectorFieldNormConcat(const VectorField3D &field) {
    long double acc = 0.0L;
    for (double v : field.x.data()) acc += static_cast<long double>(v) * v;
    for (double v : field.y.data()) acc += static_cast<long double>(v) * v;
    for (double v : field.z.data()) acc += static_cast<long double>(v) * v;
    return std::sqrt(static_cast<double>(acc));
}

LambdaUpdateResult lambdaUpdate(const VectorField3D &d,
                                const VectorField3D &d_old,
                                double lambda,
                                const VectorField3D &gv,
                                const VectorField3D &b_gu,
                                const VectorField3D &bregman,
                                int flag,
                                double lambda_init) {
    VectorField3D r_prim(b_gu.x);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(b_gu.x.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(b_gu.x.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(b_gu.x.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                r_prim.x(xi, yi, zi) = b_gu.x(xi, yi, zi) - d.x(xi, yi, zi);
                r_prim.y(xi, yi, zi) = b_gu.y(xi, yi, zi) - d.y(xi, yi, zi);
                r_prim.z(xi, yi, zi) = b_gu.z(xi, yi, zi) - d.z(xi, yi, zi);
            }
        }
    }

    const double r_prim_norm = vectorFieldNormConcat(r_prim);
    const double factor1 = vectorFieldNormConcat(b_gu);
    const double factor2 = vectorFieldNormConcat(d);
    const double r_prim_normalized = safeDivide(r_prim_norm, std::max(factor1, factor2), 0.0);

    VectorField3D d_tmp(d.x);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(d.x.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(d.x.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(d.x.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                d_tmp.x(xi, yi, zi) = d_old.x(xi, yi, zi) - d.x(xi, yi, zi);
                d_tmp.y(xi, yi, zi) = d_old.y(xi, yi, zi) - d.y(xi, yi, zi);
                d_tmp.z(xi, yi, zi) = d_old.z(xi, yi, zi) - d.z(xi, yi, zi);
            }
        }
    }

    const VectorField3D b_dd = applyPLSOperator(gv, d_tmp);
    Volume3D dual_div = sumDiagonalBackwardDivergence(b_dd);
    dual_div.scaleInPlace(lambda);
    const double r_dual_norm = dual_div.l2Norm();

    const VectorField3D b_gb = applyPLSOperator(gv, bregman);
    const Volume3D factor_div = sumDiagonalBackwardDivergence(b_gb);
    const double factor_dual = factor_div.l2Norm();
    const double r_dual_normalized = safeDivide(r_dual_norm, factor_dual, 0.0);

    const double ratio_raw = safeDivide(r_prim_normalized, r_dual_normalized, 1.0);
    const double alpha_tmp = std::sqrt(std::max(ratio_raw, std::numeric_limits<double>::epsilon()));
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

ObjectiveResult objectiveAndGradient(const Volume3D &imgy,
                                     const Volume3D &u,
                                     const VectorField3D &gv,
                                     const SeparableKernel3D &psf,
                                     const VectorField3D &d,
                                     const VectorField3D &bregman,
                                     double mu,
                                     double lambda) {
    ObjectiveResult result;
    result.gradient = Volume3D(u.nx(), u.ny(), u.nz(), 0.0);

    const Volume3D blurred = separableConvolveSame(u, psf.kx, psf.ky, psf.kz);
    Volume3D residual = blurred;
    residual.addScaledInPlace(imgy, -1.0);

    const VectorField3D gu = gradient3D(u, "forward");
    const VectorField3D b_gu = applyPLSOperator(gv, gu);
    VectorField3D fx2(b_gu.x);

    long double penalty_acc = 0.0L;
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3) reduction(+:penalty_acc)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(u.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(u.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(u.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                fx2.x(xi, yi, zi) = d.x(xi, yi, zi) - b_gu.x(xi, yi, zi) - bregman.x(xi, yi, zi);
                fx2.y(xi, yi, zi) = d.y(xi, yi, zi) - b_gu.y(xi, yi, zi) - bregman.y(xi, yi, zi);
                fx2.z(xi, yi, zi) = d.z(xi, yi, zi) - b_gu.z(xi, yi, zi) - bregman.z(xi, yi, zi);
                penalty_acc += static_cast<long double>(fx2.x(xi, yi, zi)) * fx2.x(xi, yi, zi);
                penalty_acc += static_cast<long double>(fx2.y(xi, yi, zi)) * fx2.y(xi, yi, zi);
                penalty_acc += static_cast<long double>(fx2.z(xi, yi, zi)) * fx2.z(xi, yi, zi);
            }
        }
    }

    const double residual_norm = residual.l2Norm();
    const long double data_acc = static_cast<long double>(residual_norm) * residual_norm;
    result.value = (mu / 2.0) * static_cast<double>(data_acc) + (lambda / 2.0) * static_cast<double>(penalty_acc);

    const VectorField3D b_fx2 = applyPLSOperator(gv, fx2);
    Volume3D dfx1 = separableConvolveSame(residual, psf.kx, psf.ky, psf.kz);
    Volume3D dfx2 = sumDiagonalBackwardDivergence(b_fx2);

#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(u.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(u.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(u.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                result.gradient(xi, yi, zi) = mu * dfx1(xi, yi, zi) + lambda * dfx2(xi, yi, zi);
            }
        }
    }

    return result;
}

double lineSearch(const Volume3D &imgy,
                  const Volume3D &u_current,
                  const Volume3D &u_old,
                  const VectorField3D &gv,
                  const SeparableKernel3D &psf,
                  const VectorField3D &d,
                  const VectorField3D &bregman,
                  double mu,
                  double lambda,
                  double c_value,
                  const Volume3D &gradient,
                  const SolverOptions &options) {
    Volume3D dfx_old = objectiveAndGradient(imgy, u_old, gv, psf, d, bregman, mu, lambda).gradient;

    long double s_dot_y = 0.0L;
    long double y_dot_y = 0.0L;
    const auto &x = u_current.data();
    const auto &x_old = u_old.data();
    const auto &g = gradient.data();
    const auto &g_old = dfx_old.data();
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double s = x[i] - x_old[i];
        const double y = g[i] - g_old[i];
        s_dot_y += static_cast<long double>(s) * y;
        y_dot_y += static_cast<long double>(y) * y;
    }

    double alpha = safeDivide(static_cast<double>(s_dot_y), static_cast<double>(y_dot_y), 1.0);
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        alpha = 1.0;
    }

    long double grad_sq = 0.0L;
    for (double v : g) {
        grad_sq += static_cast<long double>(v) * v;
    }

    Volume3D test_step = u_current;
    for (int i = 0; i < options.line_search_max_backtracks; ++i) {
        test_step = u_current;
        test_step.addScaledInPlace(gradient, -alpha);
        const double f_test = objectiveAndGradient(imgy, test_step, gv, psf, d, bregman, mu, lambda).value;
        if (f_test <= c_value - options.line_search_epsilon * alpha * static_cast<double>(grad_sq)) {
            return alpha;
        }
        alpha *= options.line_search_rho;
    }
    return alpha;
}

} // namespace

SolverResult runSplitBregmanPVC(const Volume3D &pet,
                                const Volume3D &guidance_image,
                                const SeparableKernel3D &psf,
                                const SolverOptions &options) {
    pet.requireSameShape(guidance_image, "runSplitBregmanPVC");
    if (pet.empty()) {
        throw std::runtime_error("Input PET volume is empty.");
    }

    const double scale_factor_pet = pet.max();
    if (scale_factor_pet <= 0.0) {
        throw std::runtime_error("PET max intensity must be positive for the reference scaling step.");
    }

    Volume3D pet_scaled = pet;
    pet_scaled.scaleInPlace(1.0 / scale_factor_pet);
    Volume3D guidance_scaled = guidance_image;
    const double guidance_max = guidance_scaled.max();
    if (guidance_max > 0.0) {
        guidance_scaled.scaleInPlace(1.0 / guidance_max);
    }

    VectorField3D d(pet);
    VectorField3D bregman(pet);
    Volume3D u = pet_scaled;
    Volume3D u_old(pet.nx(), pet.ny(), pet.nz(), 0.0);

    const VectorField3D gv = normalizeGuidanceGradient(guidance_scaled, options.mr_smooth);
    const ObjectiveResult init_obj = objectiveAndGradient(pet_scaled, pet_scaled, gv, psf, d, bregman, options.mu, options.lambda_init);
    double c_value = init_obj.value;
    double p_value = 1.0;
    double lambda = options.lambda_init;
    constexpr double eta_default = 0.995;
    (void)eta_default; // keep explicit tie to MATLAB reference naming

    int stop_counter = 0;
    int flag = 0;
    const double epsilon = options.epsilon_scale * static_cast<double>(pet.size()) /
                           static_cast<double>(181 * 210 * 181);

    SolverSummary summary{};
    summary.final_lambda = lambda;

    for (int iter = 1; iter <= options.niter; ++iter) {
        const Volume3D u_current = u;
        const ObjectiveResult current = objectiveAndGradient(pet_scaled, u_current, gv, psf, d, bregman, options.mu, lambda);
        const double alpha = lineSearch(pet_scaled, u_current, u_old, gv, psf, d, bregman,
                                        options.mu, lambda, c_value, current.gradient, options);
        u = u_current;
        u.addScaledInPlace(current.gradient, -alpha);

        const VectorField3D gu = gradient3D(u, "forward");
        const VectorField3D b_gu = applyPLSOperator(gv, gu);
        const VectorField3D d_old = d;
        d = shrinkPLS(b_gu, bregman, lambda);
        bregman = updateBregman(b_gu, d, bregman);

        const LambdaUpdateResult lambda_info = lambdaUpdate(d, d_old, lambda, gv, b_gu, bregman, flag, options.lambda_init);
        lambda = lambda_info.lambda;

        const double fx = objectiveAndGradient(pet_scaled, u, gv, psf, d, bregman, options.mu, lambda).value;
        const double p_new = options.eta * p_value + 1.0;
        c_value = (options.eta * p_value * c_value + fx) / p_new;
        p_value = p_new;
        u_old = u_current;

        Volume3D delta = u;
        delta.addScaledInPlace(u_current, -1.0);
        const double rel_update = safeDivide(delta.l2Norm(), u_current.l2Norm(), 0.0);

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

        if (stop_counter >= options.stop_counter_limit && flag <= 1) {
            ++flag;
            stop_counter = 0;
        } else if (lambda_info.primal_residual_normalized < epsilon && flag > 1) {
            summary.iterations_run = iter;
            summary.final_lambda = lambda;
            summary.final_primal_residual = lambda_info.primal_residual_normalized;
            summary.final_dual_residual = lambda_info.dual_residual_normalized;
            Volume3D out = u;
            out.scaleInPlace(scale_factor_pet);
            return {out, summary};
        }

        summary.iterations_run = iter;
        summary.final_lambda = lambda;
        summary.final_primal_residual = lambda_info.primal_residual_normalized;
        summary.final_dual_residual = lambda_info.dual_residual_normalized;
    }

    Volume3D out = u;
    out.scaleInPlace(scale_factor_pet);
    return {out, summary};
}

} // namespace pvc
