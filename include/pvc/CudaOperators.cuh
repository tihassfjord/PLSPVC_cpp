#pragma once

#include "pvc/DeviceVolume3D.cuh"

#include <vector>

namespace pvc {
namespace cuda {

// ---------------------------------------------------------------------------
// Forward-difference gradient (mirrored boundary), writes gx/gy/gz
// ---------------------------------------------------------------------------
void gradient3DForward(const DeviceVolume3D &input,
                       DeviceVolume3D &gx,
                       DeviceVolume3D &gy,
                       DeviceVolume3D &gz,
                       cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Backward-difference gradient (mirrored boundary)
// ---------------------------------------------------------------------------
void gradient3DBackward(const DeviceVolume3D &input,
                        DeviceVolume3D &gx,
                        DeviceVolume3D &gy,
                        DeviceVolume3D &gz,
                        cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// PLS projection:  out = v - g * dot(g, v)   (per-voxel)
// ---------------------------------------------------------------------------
void applyPLSOperator(const DeviceVolume3D &gvx, const DeviceVolume3D &gvy, const DeviceVolume3D &gvz,
                      const DeviceVolume3D &vx,  const DeviceVolume3D &vy,  const DeviceVolume3D &vz,
                      DeviceVolume3D &ox,         DeviceVolume3D &oy,       DeviceVolume3D &oz,
                      cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Fused shrink + Bregman update:
//   s = b_gu + bregman
//   d = shrink(s, 1/lambda)
//   bregman_new = bregman + b_gu - d
// Avoids two separate kernel launches and three extra reads/writes.
// ---------------------------------------------------------------------------
void fusedShrinkBregman(const DeviceVolume3D &b_gux, const DeviceVolume3D &b_guy, const DeviceVolume3D &b_guz,
                        DeviceVolume3D &dx,   DeviceVolume3D &dy,   DeviceVolume3D &dz,
                        DeviceVolume3D &bregx, DeviceVolume3D &bregy, DeviceVolume3D &bregz,
                        double lambda,
                        cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Diagonal backward divergence:  div = d(field_x)/dx_back + d(field_y)/dy_back + d(field_z)/dz_back
// Fused: computes three backward gradients and sums the diagonal elements in one kernel.
// ---------------------------------------------------------------------------
void sumDiagonalBackwardDivergence(const DeviceVolume3D &fx, const DeviceVolume3D &fy, const DeviceVolume3D &fz,
                                   DeviceVolume3D &out,
                                   cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Separable 3-D convolution (same-size output, zero-pad boundary).
// Uses shared memory tiling for the inner-axis pass.
// `tmp1` and `tmp2` are pre-allocated scratch volumes of the same size.
// ---------------------------------------------------------------------------
void separableConvolve3D(const DeviceVolume3D &input,
                         const double *d_kx, int kx_len,
                         const double *d_ky, int ky_len,
                         const double *d_kz, int kz_len,
                         DeviceVolume3D &tmp1,
                         DeviceVolume3D &tmp2,
                         DeviceVolume3D &output,
                         cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Normalize guidance gradient in-place: g /= sqrt(gx²+gy²+gz²+eps²)
// ---------------------------------------------------------------------------
void normalizeGuidanceGradient(DeviceVolume3D &gvx, DeviceVolume3D &gvy, DeviceVolume3D &gvz,
                               double mr_smooth,
                               cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Element-wise: out = a + alpha * b
// ---------------------------------------------------------------------------
void addScaled(const DeviceVolume3D &a, const DeviceVolume3D &b, double alpha,
               DeviceVolume3D &out,
               cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Element-wise: dst *= scale
// ---------------------------------------------------------------------------
void scaleInPlace(DeviceVolume3D &dst, double scale, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Fused objective-gradient kernel pieces
// ---------------------------------------------------------------------------

// Computes fx2 = d - b_gu - bregman  AND  accumulates penalty = sum(fx2²)
// Returns the partial-sum buffer (use thrust/cub to finish the reduction).
void computeFx2AndPenalty(const DeviceVolume3D &dx, const DeviceVolume3D &dy, const DeviceVolume3D &dz,
                          const DeviceVolume3D &b_gux, const DeviceVolume3D &b_guy, const DeviceVolume3D &b_guz,
                          const DeviceVolume3D &bregx, const DeviceVolume3D &bregy, const DeviceVolume3D &bregz,
                          DeviceVolume3D &fx2x, DeviceVolume3D &fx2y, DeviceVolume3D &fx2z,
                          double *d_penalty_sum,
                          cudaStream_t stream = 0);

// Fused gradient combine:  grad = mu * dfx1 + lambda * dfx2
void fusedGradientCombine(const DeviceVolume3D &dfx1, const DeviceVolume3D &dfx2,
                          double mu, double lambda,
                          DeviceVolume3D &grad,
                          cudaStream_t stream = 0);

// Primal residual:  r_prim = b_gu - d  (writes into out vector field)
void computePrimalResidual(const DeviceVolume3D &b_gux, const DeviceVolume3D &b_guy, const DeviceVolume3D &b_guz,
                           const DeviceVolume3D &dx, const DeviceVolume3D &dy, const DeviceVolume3D &dz,
                           DeviceVolume3D &rx, DeviceVolume3D &ry, DeviceVolume3D &rz,
                           cudaStream_t stream = 0);

// d_tmp = d_old - d  (element-wise vector field subtraction)
void vectorFieldSubtract(const DeviceVolume3D &ax, const DeviceVolume3D &ay, const DeviceVolume3D &az,
                         const DeviceVolume3D &bx, const DeviceVolume3D &by, const DeviceVolume3D &bz,
                         DeviceVolume3D &ox, DeviceVolume3D &oy, DeviceVolume3D &oz,
                         cudaStream_t stream = 0);

// Fused line-search helpers:
//   s_dot_y = sum( (x-x_old) * (g-g_old) )
//   y_dot_y = sum( (g-g_old)^2 )
//   grad_sq = sum( g^2 )
void lineSearchDots(const DeviceVolume3D &x, const DeviceVolume3D &x_old,
                    const DeviceVolume3D &g, const DeviceVolume3D &g_old,
                    double *d_s_dot_y, double *d_y_dot_y, double *d_grad_sq,
                    cudaStream_t stream = 0);

} // namespace cuda
} // namespace pvc
