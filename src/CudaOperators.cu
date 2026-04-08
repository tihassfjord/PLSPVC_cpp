#include "pvc/CudaOperators.cuh"

#include <cmath>
#include <limits>

#include <cub/cub.cuh>

namespace pvc {
namespace cuda {

// ===== Launch helpers ======================================================

static constexpr int kBlockSize = 256;

static inline int gridSize(std::size_t n) {
    return static_cast<int>((n + kBlockSize - 1) / kBlockSize);
}

// Flat index helpers (row-major: (z*ny+y)*nx+x)
__device__ __forceinline__ std::size_t idx3(std::size_t x, std::size_t y, std::size_t z,
                                            std::size_t nx, std::size_t ny) {
    return (z * ny + y) * nx + x;
}

// ===== gradient3DForward ===================================================

__global__ void k_gradient3DForward(const double * __restrict__ in,
                                     double * __restrict__ gx,
                                     double * __restrict__ gy,
                                     double * __restrict__ gz,
                                     std::size_t nx, std::size_t ny, std::size_t nz)
{
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t N = nx * ny * nz;
    if (tid >= N) return;

    const std::size_t x = tid % nx;
    const std::size_t y = (tid / nx) % ny;
    const std::size_t z = tid / (nx * ny);

    const double c = in[tid];

    // X forward diff with mirror
    if (nx <= 1) {
        gx[tid] = 0.0;
    } else if (x + 1 < nx) {
        gx[tid] = in[idx3(x + 1, y, z, nx, ny)] - c;
    } else {
        gx[tid] = in[idx3(nx - 2, y, z, nx, ny)] - c;
    }

    // Y
    if (ny <= 1) {
        gy[tid] = 0.0;
    } else if (y + 1 < ny) {
        gy[tid] = in[idx3(x, y + 1, z, nx, ny)] - c;
    } else {
        gy[tid] = in[idx3(x, ny - 2, z, nx, ny)] - c;
    }

    // Z
    if (nz <= 1) {
        gz[tid] = 0.0;
    } else if (z + 1 < nz) {
        gz[tid] = in[idx3(x, y, z + 1, nx, ny)] - c;
    } else {
        gz[tid] = in[idx3(x, y, nz - 2, nx, ny)] - c;
    }
}

void gradient3DForward(const DeviceVolume3D &input,
                       DeviceVolume3D &gx, DeviceVolume3D &gy, DeviceVolume3D &gz,
                       cudaStream_t stream) {
    const std::size_t N = input.size();
    k_gradient3DForward<<<gridSize(N), kBlockSize, 0, stream>>>(
        input.ptr, gx.ptr, gy.ptr, gz.ptr, input.nx, input.ny, input.nz);
}

// ===== gradient3DBackward ==================================================

__global__ void k_gradient3DBackward(const double * __restrict__ in,
                                      double * __restrict__ gx,
                                      double * __restrict__ gy,
                                      double * __restrict__ gz,
                                      std::size_t nx, std::size_t ny, std::size_t nz)
{
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t N = nx * ny * nz;
    if (tid >= N) return;

    const std::size_t x = tid % nx;
    const std::size_t y = (tid / nx) % ny;
    const std::size_t z = tid / (nx * ny);

    const double c = in[tid];

    if (nx <= 1) {
        gx[tid] = 0.0;
    } else if (x > 0) {
        gx[tid] = c - in[idx3(x - 1, y, z, nx, ny)];
    } else {
        gx[tid] = c - in[idx3(1, y, z, nx, ny)];
    }

    if (ny <= 1) {
        gy[tid] = 0.0;
    } else if (y > 0) {
        gy[tid] = c - in[idx3(x, y - 1, z, nx, ny)];
    } else {
        gy[tid] = c - in[idx3(x, 1, z, nx, ny)];
    }

    if (nz <= 1) {
        gz[tid] = 0.0;
    } else if (z > 0) {
        gz[tid] = c - in[idx3(x, y, z - 1, nx, ny)];
    } else {
        gz[tid] = c - in[idx3(x, y, 1, nx, ny)];
    }
}

void gradient3DBackward(const DeviceVolume3D &input,
                        DeviceVolume3D &gx, DeviceVolume3D &gy, DeviceVolume3D &gz,
                        cudaStream_t stream) {
    const std::size_t N = input.size();
    k_gradient3DBackward<<<gridSize(N), kBlockSize, 0, stream>>>(
        input.ptr, gx.ptr, gy.ptr, gz.ptr, input.nx, input.ny, input.nz);
}

// ===== applyPLSOperator ====================================================

__global__ void k_applyPLS(const double * __restrict__ gvx, const double * __restrict__ gvy, const double * __restrict__ gvz,
                            const double * __restrict__ vx,  const double * __restrict__ vy,  const double * __restrict__ vz,
                            double * __restrict__ ox,        double * __restrict__ oy,        double * __restrict__ oz,
                            std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    const double gx = gvx[i], gy = gvy[i], gz = gvz[i];
    const double ix = vx[i],  iy = vy[i],  iz = vz[i];
    const double dot = gx * ix + gy * iy + gz * iz;
    ox[i] = ix - gx * dot;
    oy[i] = iy - gy * dot;
    oz[i] = iz - gz * dot;
}

void applyPLSOperator(const DeviceVolume3D &gvx, const DeviceVolume3D &gvy, const DeviceVolume3D &gvz,
                      const DeviceVolume3D &vx,  const DeviceVolume3D &vy,  const DeviceVolume3D &vz,
                      DeviceVolume3D &ox,         DeviceVolume3D &oy,       DeviceVolume3D &oz,
                      cudaStream_t stream) {
    const std::size_t N = gvx.size();
    k_applyPLS<<<gridSize(N), kBlockSize, 0, stream>>>(
        gvx.ptr, gvy.ptr, gvz.ptr, vx.ptr, vy.ptr, vz.ptr, ox.ptr, oy.ptr, oz.ptr, N);
}

// ===== fusedShrinkBregman ==================================================
// Fuses: shrinkPLS + updateBregman into a single kernel pass.

__global__ void k_fusedShrinkBregman(const double * __restrict__ b_gux, const double * __restrict__ b_guy, const double * __restrict__ b_guz,
                                      double * __restrict__ dx,   double * __restrict__ dy,   double * __restrict__ dz,
                                      double * __restrict__ bregx, double * __restrict__ bregy, double * __restrict__ bregz,
                                      double inv_lambda, std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;

    const double bx = bregx[i], by = bregy[i], bz = bregz[i];
    const double sx = b_gux[i] + bx;
    const double sy = b_guy[i] + by;
    const double sz = b_guz[i] + bz;

    const double s = sqrt(sx * sx + sy * sy + sz * sz);
    const double tiny = 1e-300;
    const double tmp = fmax(s - inv_lambda, 0.0);
    const double scale = tmp / (s + tiny);

    const double new_dx = scale * sx;
    const double new_dy = scale * sy;
    const double new_dz = scale * sz;
    dx[i] = new_dx;
    dy[i] = new_dy;
    dz[i] = new_dz;

    // Bregman update: bregman_new = bregman + b_gu - d
    bregx[i] = bx + b_gux[i] - new_dx;
    bregy[i] = by + b_guy[i] - new_dy;
    bregz[i] = bz + b_guz[i] - new_dz;
}

void fusedShrinkBregman(const DeviceVolume3D &b_gux, const DeviceVolume3D &b_guy, const DeviceVolume3D &b_guz,
                        DeviceVolume3D &dx, DeviceVolume3D &dy, DeviceVolume3D &dz,
                        DeviceVolume3D &bregx, DeviceVolume3D &bregy, DeviceVolume3D &bregz,
                        double lambda, cudaStream_t stream) {
    const std::size_t N = b_gux.size();
    const double inv_lambda = 1.0 / lambda;
    k_fusedShrinkBregman<<<gridSize(N), kBlockSize, 0, stream>>>(
        b_gux.ptr, b_guy.ptr, b_guz.ptr,
        dx.ptr, dy.ptr, dz.ptr,
        bregx.ptr, bregy.ptr, bregz.ptr,
        inv_lambda, N);
}

// ===== sumDiagonalBackwardDivergence =======================================
// Fused kernel: computes backward differences of each vector component along
// its own axis and sums the diagonals, avoiding three separate gradient calls.

__global__ void k_sumDiagBackDiv(const double * __restrict__ fx,
                                  const double * __restrict__ fy,
                                  const double * __restrict__ fz,
                                  double * __restrict__ out,
                                  std::size_t nx, std::size_t ny, std::size_t nz)
{
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t N = nx * ny * nz;
    if (tid >= N) return;

    const std::size_t x = tid % nx;
    const std::size_t y = (tid / nx) % ny;
    const std::size_t z = tid / (nx * ny);

    // d(fx)/dx backward
    double div_x;
    if (nx <= 1) { div_x = 0.0; }
    else if (x > 0) { div_x = fx[tid] - fx[idx3(x - 1, y, z, nx, ny)]; }
    else { div_x = fx[tid] - fx[idx3(1, y, z, nx, ny)]; }

    // d(fy)/dy backward
    double div_y;
    if (ny <= 1) { div_y = 0.0; }
    else if (y > 0) { div_y = fy[tid] - fy[idx3(x, y - 1, z, nx, ny)]; }
    else { div_y = fy[tid] - fy[idx3(x, 1, z, nx, ny)]; }

    // d(fz)/dz backward
    double div_z;
    if (nz <= 1) { div_z = 0.0; }
    else if (z > 0) { div_z = fz[tid] - fz[idx3(x, y, z - 1, nx, ny)]; }
    else { div_z = fz[tid] - fz[idx3(x, y, 1, nx, ny)]; }

    out[tid] = div_x + div_y + div_z;
}

void sumDiagonalBackwardDivergence(const DeviceVolume3D &fx, const DeviceVolume3D &fy, const DeviceVolume3D &fz,
                                   DeviceVolume3D &out, cudaStream_t stream) {
    const std::size_t N = fx.size();
    k_sumDiagBackDiv<<<gridSize(N), kBlockSize, 0, stream>>>(
        fx.ptr, fy.ptr, fz.ptr, out.ptr, fx.nx, fx.ny, fx.nz);
}

// ===== Separable 3-D convolution ===========================================
// Three 1-D passes along X, Y, Z.  Uses constant memory for kernel weights
// (max 64 taps per axis – more than enough for any practical PSF).

static constexpr int kMaxKernelLen = 64;
__constant__ double c_kx[kMaxKernelLen];
__constant__ double c_ky[kMaxKernelLen];
__constant__ double c_kz[kMaxKernelLen];

// Generic 1-D convolution along a specified axis
__global__ void k_conv1D_X(const double * __restrict__ in,
                            double * __restrict__ out,
                            std::size_t nx, std::size_t ny, std::size_t nz,
                            int radius)
{
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t N = nx * ny * nz;
    if (tid >= N) return;

    const std::size_t x = tid % nx;
    const std::size_t y = (tid / nx) % ny;
    const std::size_t z = tid / (nx * ny);

    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        std::ptrdiff_t xx = static_cast<std::ptrdiff_t>(x) + k;
        if (xx >= 0 && xx < static_cast<std::ptrdiff_t>(nx)) {
            sum += c_kx[k + radius] * in[idx3(static_cast<std::size_t>(xx), y, z, nx, ny)];
        }
    }
    out[tid] = sum;
}

__global__ void k_conv1D_Y(const double * __restrict__ in,
                            double * __restrict__ out,
                            std::size_t nx, std::size_t ny, std::size_t nz,
                            int radius)
{
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t N = nx * ny * nz;
    if (tid >= N) return;

    const std::size_t x = tid % nx;
    const std::size_t y = (tid / nx) % ny;
    const std::size_t z = tid / (nx * ny);

    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        std::ptrdiff_t yy = static_cast<std::ptrdiff_t>(y) + k;
        if (yy >= 0 && yy < static_cast<std::ptrdiff_t>(ny)) {
            sum += c_ky[k + radius] * in[idx3(x, static_cast<std::size_t>(yy), z, nx, ny)];
        }
    }
    out[tid] = sum;
}

__global__ void k_conv1D_Z(const double * __restrict__ in,
                            double * __restrict__ out,
                            std::size_t nx, std::size_t ny, std::size_t nz,
                            int radius)
{
    const std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t N = nx * ny * nz;
    if (tid >= N) return;

    const std::size_t x = tid % nx;
    const std::size_t y = (tid / nx) % ny;
    const std::size_t z = tid / (nx * ny);

    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        std::ptrdiff_t zz = static_cast<std::ptrdiff_t>(z) + k;
        if (zz >= 0 && zz < static_cast<std::ptrdiff_t>(nz)) {
            sum += c_kz[k + radius] * in[idx3(x, y, static_cast<std::size_t>(zz), nx, ny)];
        }
    }
    out[tid] = sum;
}

void separableConvolve3D(const DeviceVolume3D &input,
                         const double *h_kx, int kx_len,
                         const double *h_ky, int ky_len,
                         const double *h_kz, int kz_len,
                         DeviceVolume3D &tmp1,
                         DeviceVolume3D &tmp2,
                         DeviceVolume3D &output,
                         cudaStream_t stream) {
    const std::size_t N = input.size();
    const int g = gridSize(N);

    // Copy kernel weights to constant memory
    cudaMemcpyToSymbolAsync(c_kx, h_kx, kx_len * sizeof(double), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_ky, h_ky, ky_len * sizeof(double), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_kz, h_kz, kz_len * sizeof(double), 0, cudaMemcpyHostToDevice, stream);

    const int rx = kx_len / 2;
    const int ry = ky_len / 2;
    const int rz = kz_len / 2;

    // X → tmp1,  Y → tmp2,  Z → output
    k_conv1D_X<<<g, kBlockSize, 0, stream>>>(input.ptr, tmp1.ptr, input.nx, input.ny, input.nz, rx);
    k_conv1D_Y<<<g, kBlockSize, 0, stream>>>(tmp1.ptr,  tmp2.ptr, input.nx, input.ny, input.nz, ry);
    k_conv1D_Z<<<g, kBlockSize, 0, stream>>>(tmp2.ptr,  output.ptr, input.nx, input.ny, input.nz, rz);
}

// ===== normalizeGuidanceGradient ===========================================

__global__ void k_normalizeGV(double * __restrict__ gvx,
                               double * __restrict__ gvy,
                               double * __restrict__ gvz,
                               double mr_smooth_sq, std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    const double x = gvx[i], y = gvy[i], z = gvz[i];
    const double norm = sqrt(x * x + y * y + z * z + mr_smooth_sq);
    gvx[i] = x / norm;
    gvy[i] = y / norm;
    gvz[i] = z / norm;
}

void normalizeGuidanceGradient(DeviceVolume3D &gvx, DeviceVolume3D &gvy, DeviceVolume3D &gvz,
                               double mr_smooth, cudaStream_t stream) {
    const std::size_t N = gvx.size();
    k_normalizeGV<<<gridSize(N), kBlockSize, 0, stream>>>(
        gvx.ptr, gvy.ptr, gvz.ptr, mr_smooth * mr_smooth, N);
}

// ===== addScaled ===========================================================

__global__ void k_addScaled(const double * __restrict__ a, const double * __restrict__ b,
                             double alpha, double * __restrict__ out, std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] + alpha * b[i];
}

void addScaled(const DeviceVolume3D &a, const DeviceVolume3D &b, double alpha,
               DeviceVolume3D &out, cudaStream_t stream) {
    const std::size_t N = a.size();
    k_addScaled<<<gridSize(N), kBlockSize, 0, stream>>>(a.ptr, b.ptr, alpha, out.ptr, N);
}

// ===== scaleInPlace ========================================================

__global__ void k_scaleInPlace(double * __restrict__ d, double s, std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    d[i] *= s;
}

void scaleInPlace(DeviceVolume3D &dst, double scale, cudaStream_t stream) {
    const std::size_t N = dst.size();
    k_scaleInPlace<<<gridSize(N), kBlockSize, 0, stream>>>(dst.ptr, scale, N);
}

// ===== computeFx2AndPenalty ================================================
// Fused: fx2 = d - b_gu - bregman, penalty = sum(fx2²)

__global__ void k_fx2Penalty(const double * __restrict__ dx, const double * __restrict__ dy, const double * __restrict__ dz,
                              const double * __restrict__ bx, const double * __restrict__ by, const double * __restrict__ bz,
                              const double * __restrict__ brx, const double * __restrict__ bry, const double * __restrict__ brz,
                              double * __restrict__ fx2x, double * __restrict__ fx2y, double * __restrict__ fx2z,
                              double * __restrict__ penalty,
                              std::size_t N)
{
    // Use CUB block-level reduction for the penalty sum
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;

    double local_sum = 0.0;
    if (i < N) {
        const double ex = dx[i] - bx[i] - brx[i];
        const double ey = dy[i] - by[i] - bry[i];
        const double ez = dz[i] - bz[i] - brz[i];
        fx2x[i] = ex;
        fx2y[i] = ey;
        fx2z[i] = ez;
        local_sum = ex * ex + ey * ey + ez * ez;
    }

    double block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(penalty, block_sum);
    }
}

void computeFx2AndPenalty(const DeviceVolume3D &dx, const DeviceVolume3D &dy, const DeviceVolume3D &dz,
                          const DeviceVolume3D &b_gux, const DeviceVolume3D &b_guy, const DeviceVolume3D &b_guz,
                          const DeviceVolume3D &bregx, const DeviceVolume3D &bregy, const DeviceVolume3D &bregz,
                          DeviceVolume3D &fx2x, DeviceVolume3D &fx2y, DeviceVolume3D &fx2z,
                          double *d_penalty_sum, cudaStream_t stream) {
    // Zero the accumulator
    cudaMemsetAsync(d_penalty_sum, 0, sizeof(double), stream);

    const std::size_t N = dx.size();
    k_fx2Penalty<<<gridSize(N), kBlockSize, 0, stream>>>(
        dx.ptr, dy.ptr, dz.ptr,
        b_gux.ptr, b_guy.ptr, b_guz.ptr,
        bregx.ptr, bregy.ptr, bregz.ptr,
        fx2x.ptr, fx2y.ptr, fx2z.ptr,
        d_penalty_sum, N);
}

// ===== fusedGradientCombine ================================================

__global__ void k_gradCombine(const double * __restrict__ dfx1,
                               const double * __restrict__ dfx2,
                               double mu, double lam,
                               double * __restrict__ grad, std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    grad[i] = mu * dfx1[i] + lam * dfx2[i];
}

void fusedGradientCombine(const DeviceVolume3D &dfx1, const DeviceVolume3D &dfx2,
                          double mu, double lambda,
                          DeviceVolume3D &grad, cudaStream_t stream) {
    const std::size_t N = dfx1.size();
    k_gradCombine<<<gridSize(N), kBlockSize, 0, stream>>>(
        dfx1.ptr, dfx2.ptr, mu, lambda, grad.ptr, N);
}

// ===== computePrimalResidual ===============================================

__global__ void k_vfSubtract(const double * __restrict__ ax, const double * __restrict__ ay, const double * __restrict__ az,
                              const double * __restrict__ bx, const double * __restrict__ by, const double * __restrict__ bz,
                              double * __restrict__ ox, double * __restrict__ oy, double * __restrict__ oz,
                              std::size_t N)
{
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    ox[i] = ax[i] - bx[i];
    oy[i] = ay[i] - by[i];
    oz[i] = az[i] - bz[i];
}

void computePrimalResidual(const DeviceVolume3D &b_gux, const DeviceVolume3D &b_guy, const DeviceVolume3D &b_guz,
                           const DeviceVolume3D &dx, const DeviceVolume3D &dy, const DeviceVolume3D &dz,
                           DeviceVolume3D &rx, DeviceVolume3D &ry, DeviceVolume3D &rz,
                           cudaStream_t stream) {
    const std::size_t N = b_gux.size();
    k_vfSubtract<<<gridSize(N), kBlockSize, 0, stream>>>(
        b_gux.ptr, b_guy.ptr, b_guz.ptr,
        dx.ptr, dy.ptr, dz.ptr,
        rx.ptr, ry.ptr, rz.ptr, N);
}

void vectorFieldSubtract(const DeviceVolume3D &ax, const DeviceVolume3D &ay, const DeviceVolume3D &az,
                         const DeviceVolume3D &bx, const DeviceVolume3D &by, const DeviceVolume3D &bz,
                         DeviceVolume3D &ox, DeviceVolume3D &oy, DeviceVolume3D &oz,
                         cudaStream_t stream) {
    const std::size_t N = ax.size();
    k_vfSubtract<<<gridSize(N), kBlockSize, 0, stream>>>(
        ax.ptr, ay.ptr, az.ptr, bx.ptr, by.ptr, bz.ptr, ox.ptr, oy.ptr, oz.ptr, N);
}

// ===== lineSearchDots ======================================================

__global__ void k_lineSearchDots(const double * __restrict__ x_ptr,
                                  const double * __restrict__ x_old_ptr,
                                  const double * __restrict__ g_ptr,
                                  const double * __restrict__ g_old_ptr,
                                  double * __restrict__ d_sdy,
                                  double * __restrict__ d_ydy,
                                  double * __restrict__ d_gsq,
                                  std::size_t N)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage ts1, ts2, ts3;

    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;

    double sdy = 0.0, ydy = 0.0, gsq = 0.0;
    if (i < N) {
        const double s = x_ptr[i] - x_old_ptr[i];
        const double y = g_ptr[i] - g_old_ptr[i];
        sdy = s * y;
        ydy = y * y;
        gsq = g_ptr[i] * g_ptr[i];
    }

    double b1 = BlockReduce(ts1).Sum(sdy);
    double b2 = BlockReduce(ts2).Sum(ydy);
    double b3 = BlockReduce(ts3).Sum(gsq);
    if (threadIdx.x == 0) {
        atomicAdd(d_sdy, b1);
        atomicAdd(d_ydy, b2);
        atomicAdd(d_gsq, b3);
    }
}

void lineSearchDots(const DeviceVolume3D &x, const DeviceVolume3D &x_old,
                    const DeviceVolume3D &g, const DeviceVolume3D &g_old,
                    double *d_s_dot_y, double *d_y_dot_y, double *d_grad_sq,
                    cudaStream_t stream) {
    cudaMemsetAsync(d_s_dot_y, 0, sizeof(double), stream);
    cudaMemsetAsync(d_y_dot_y, 0, sizeof(double), stream);
    cudaMemsetAsync(d_grad_sq, 0, sizeof(double), stream);

    const std::size_t N = x.size();
    k_lineSearchDots<<<gridSize(N), kBlockSize, 0, stream>>>(
        x.ptr, x_old.ptr, g.ptr, g_old.ptr,
        d_s_dot_y, d_y_dot_y, d_grad_sq, N);
}

} // namespace cuda
} // namespace pvc
