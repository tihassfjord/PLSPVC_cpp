#include "pvc/Operators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvc {
namespace {

inline double mirroredForwardDifference(const Volume3D &x, std::size_t i, std::size_t j, std::size_t k, int axis) {
    auto get = [&](std::size_t a, std::size_t b, std::size_t c) { return x(a, b, c); };
    switch (axis) {
    case 0:
        if (x.nx() <= 1) return 0.0;
        return (i + 1 < x.nx()) ? get(i + 1, j, k) - get(i, j, k) : get(x.nx() - 2, j, k) - get(x.nx() - 1, j, k);
    case 1:
        if (x.ny() <= 1) return 0.0;
        return (j + 1 < x.ny()) ? get(i, j + 1, k) - get(i, j, k) : get(i, x.ny() - 2, k) - get(i, x.ny() - 1, k);
    case 2:
        if (x.nz() <= 1) return 0.0;
        return (k + 1 < x.nz()) ? get(i, j, k + 1) - get(i, j, k) : get(i, j, x.nz() - 2) - get(i, j, x.nz() - 1);
    default:
        return 0.0;
    }
}

inline double mirroredBackwardDifference(const Volume3D &x, std::size_t i, std::size_t j, std::size_t k, int axis) {
    auto get = [&](std::size_t a, std::size_t b, std::size_t c) { return x(a, b, c); };
    switch (axis) {
    case 0:
        if (x.nx() <= 1) return 0.0;
        return (i > 0) ? get(i, j, k) - get(i - 1, j, k) : get(0, j, k) - get(1, j, k);
    case 1:
        if (x.ny() <= 1) return 0.0;
        return (j > 0) ? get(i, j, k) - get(i, j - 1, k) : get(i, 0, k) - get(i, 1, k);
    case 2:
        if (x.nz() <= 1) return 0.0;
        return (k > 0) ? get(i, j, k) - get(i, j, k - 1) : get(i, j, 0) - get(i, j, 1);
    default:
        return 0.0;
    }
}

Volume3D convolve1DAlongAxis(const Volume3D &input, const std::vector<double> &kernel, int axis) {
    const int radius = static_cast<int>(kernel.size() / 2);
    Volume3D out(input.nx(), input.ny(), input.nz(), 0.0);

#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(input.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(input.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(input.nx()); ++x) {
                double sum = 0.0;
                for (int kk = -radius; kk <= radius; ++kk) {
                    std::ptrdiff_t xx = x;
                    std::ptrdiff_t yy = y;
                    std::ptrdiff_t zz = z;
                    if (axis == 0) xx += kk;
                    if (axis == 1) yy += kk;
                    if (axis == 2) zz += kk;
                    if (xx < 0 || yy < 0 || zz < 0 ||
                        xx >= static_cast<std::ptrdiff_t>(input.nx()) ||
                        yy >= static_cast<std::ptrdiff_t>(input.ny()) ||
                        zz >= static_cast<std::ptrdiff_t>(input.nz())) {
                        continue;
                    }
                    sum += kernel[static_cast<std::size_t>(kk + radius)] * input(static_cast<std::size_t>(xx), static_cast<std::size_t>(yy), static_cast<std::size_t>(zz));
                }
                out(static_cast<std::size_t>(x), static_cast<std::size_t>(y), static_cast<std::size_t>(z)) = sum;
            }
        }
    }

    return out;
}

} // namespace

VectorField3D gradient3D(const Volume3D &input, const std::string &mode) {
    if (mode != "forward" && mode != "backward") {
        throw std::runtime_error("gradient3D mode must be either 'forward' or 'backward'.");
    }

    VectorField3D out(input);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(input.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(input.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(input.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                if (mode == "forward") {
                    out.x(xi, yi, zi) = mirroredForwardDifference(input, xi, yi, zi, 0);
                    out.y(xi, yi, zi) = mirroredForwardDifference(input, xi, yi, zi, 1);
                    out.z(xi, yi, zi) = mirroredForwardDifference(input, xi, yi, zi, 2);
                } else {
                    out.x(xi, yi, zi) = mirroredBackwardDifference(input, xi, yi, zi, 0);
                    out.y(xi, yi, zi) = mirroredBackwardDifference(input, xi, yi, zi, 1);
                    out.z(xi, yi, zi) = mirroredBackwardDifference(input, xi, yi, zi, 2);
                }
            }
        }
    }
    return out;
}

VectorField3D applyPLSOperator(const VectorField3D &guidance_gradient, const VectorField3D &input) {
    guidance_gradient.x.requireSameShape(input.x, "applyPLSOperator.x");
    guidance_gradient.y.requireSameShape(input.y, "applyPLSOperator.y");
    guidance_gradient.z.requireSameShape(input.z, "applyPLSOperator.z");

    VectorField3D out(input.x);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(input.x.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(input.x.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(input.x.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                const double gx = guidance_gradient.x(xi, yi, zi);
                const double gy = guidance_gradient.y(xi, yi, zi);
                const double gz = guidance_gradient.z(xi, yi, zi);
                const double vx = input.x(xi, yi, zi);
                const double vy = input.y(xi, yi, zi);
                const double vz = input.z(xi, yi, zi);
                const double dot = gx * vx + gy * vy + gz * vz;
                out.x(xi, yi, zi) = vx - gx * dot;
                out.y(xi, yi, zi) = vy - gy * dot;
                out.z(xi, yi, zi) = vz - gz * dot;
            }
        }
    }
    return out;
}

VectorField3D shrinkPLS(const VectorField3D &b_gu, const VectorField3D &bregman, double lambda) {
    VectorField3D out(b_gu.x);
    const double tiny = std::numeric_limits<double>::epsilon();
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(b_gu.x.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(b_gu.x.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(b_gu.x.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                const double sx = b_gu.x(xi, yi, zi) + bregman.x(xi, yi, zi);
                const double sy = b_gu.y(xi, yi, zi) + bregman.y(xi, yi, zi);
                const double sz = b_gu.z(xi, yi, zi) + bregman.z(xi, yi, zi);
                const double s = std::sqrt(sx * sx + sy * sy + sz * sz);
                const double tmp = std::max(s - 1.0 / lambda, 0.0);
                const double scale = tmp / (s + tiny);
                out.x(xi, yi, zi) = scale * sx;
                out.y(xi, yi, zi) = scale * sy;
                out.z(xi, yi, zi) = scale * sz;
            }
        }
    }
    return out;
}

VectorField3D updateBregman(const VectorField3D &b_gu, const VectorField3D &d, const VectorField3D &bregman) {
    VectorField3D out(b_gu.x);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(b_gu.x.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(b_gu.x.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(b_gu.x.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                out.x(xi, yi, zi) = bregman.x(xi, yi, zi) + b_gu.x(xi, yi, zi) - d.x(xi, yi, zi);
                out.y(xi, yi, zi) = bregman.y(xi, yi, zi) + b_gu.y(xi, yi, zi) - d.y(xi, yi, zi);
                out.z(xi, yi, zi) = bregman.z(xi, yi, zi) + b_gu.z(xi, yi, zi) - d.z(xi, yi, zi);
            }
        }
    }
    return out;
}

Volume3D sumDiagonalBackwardDivergence(const VectorField3D &field) {
    const VectorField3D bx = gradient3D(field.x, "backward");
    const VectorField3D by = gradient3D(field.y, "backward");
    const VectorField3D bz = gradient3D(field.z, "backward");
    Volume3D out(field.x.nx(), field.x.ny(), field.x.nz(), 0.0);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(out.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(out.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(out.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                out(xi, yi, zi) = bx.x(xi, yi, zi) + by.y(xi, yi, zi) + bz.z(xi, yi, zi);
            }
        }
    }
    return out;
}

Volume3D convolveSame(const Volume3D &input, const Volume3D &kernel) {
    const int cx = static_cast<int>(kernel.nx() / 2);
    const int cy = static_cast<int>(kernel.ny() / 2);
    const int cz = static_cast<int>(kernel.nz() / 2);
    Volume3D out(input.nx(), input.ny(), input.nz(), 0.0);

#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(input.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(input.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(input.nx()); ++x) {
                double sum = 0.0;
                for (std::ptrdiff_t kz = 0; kz < static_cast<std::ptrdiff_t>(kernel.nz()); ++kz) {
                    for (std::ptrdiff_t ky = 0; ky < static_cast<std::ptrdiff_t>(kernel.ny()); ++ky) {
                        for (std::ptrdiff_t kx = 0; kx < static_cast<std::ptrdiff_t>(kernel.nx()); ++kx) {
                            const std::ptrdiff_t xx = x + (kx - cx);
                            const std::ptrdiff_t yy = y + (ky - cy);
                            const std::ptrdiff_t zz = z + (kz - cz);
                            if (xx < 0 || yy < 0 || zz < 0 ||
                                xx >= static_cast<std::ptrdiff_t>(input.nx()) ||
                                yy >= static_cast<std::ptrdiff_t>(input.ny()) ||
                                zz >= static_cast<std::ptrdiff_t>(input.nz())) {
                                continue;
                            }
                            sum += kernel(static_cast<std::size_t>(kx), static_cast<std::size_t>(ky), static_cast<std::size_t>(kz)) *
                                   input(static_cast<std::size_t>(xx), static_cast<std::size_t>(yy), static_cast<std::size_t>(zz));
                        }
                    }
                }
                out(static_cast<std::size_t>(x), static_cast<std::size_t>(y), static_cast<std::size_t>(z)) = sum;
            }
        }
    }
    return out;
}

Volume3D separableConvolveSame(const Volume3D &input,
                               const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz) {
    Volume3D tmp_x = convolve1DAlongAxis(input, kx, 0);
    Volume3D tmp_y = convolve1DAlongAxis(tmp_x, ky, 1);
    Volume3D tmp_z = convolve1DAlongAxis(tmp_y, kz, 2);
    return tmp_z;
}

Volume3D flipKernel(const Volume3D &kernel) {
    Volume3D out(kernel.nx(), kernel.ny(), kernel.nz(), 0.0);
    for (std::size_t z = 0; z < kernel.nz(); ++z) {
        for (std::size_t y = 0; y < kernel.ny(); ++y) {
            for (std::size_t x = 0; x < kernel.nx(); ++x) {
                out(x, y, z) = kernel(kernel.nx() - 1 - x, kernel.ny() - 1 - y, kernel.nz() - 1 - z);
            }
        }
    }
    return out;
}

Volume3D gradientMagnitudeImage(const Volume3D &input) {
    const VectorField3D g = gradient3D(input, "forward");
    Volume3D out(input.nx(), input.ny(), input.nz(), 0.0);
#ifdef PVC_USE_OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(input.nz()); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(input.ny()); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(input.nx()); ++x) {
                const std::size_t xi = static_cast<std::size_t>(x);
                const std::size_t yi = static_cast<std::size_t>(y);
                const std::size_t zi = static_cast<std::size_t>(z);
                const double gx = g.x(xi, yi, zi);
                const double gy = g.y(xi, yi, zi);
                const double gz = g.z(xi, yi, zi);
                out(xi, yi, zi) = std::sqrt(gx * gx + gy * gy + gz * gz);
            }
        }
    }
    return out;
}

Volume3D normalizeToUnitMax(const Volume3D &input) {
    Volume3D out = input;
    const double max_value = out.max();
    if (max_value > 0.0) {
        out.scaleInPlace(1.0 / max_value);
    }
    return out;
}

} // namespace pvc
