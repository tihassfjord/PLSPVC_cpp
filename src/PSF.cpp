#include "pvc/PSF.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pvc {
namespace {

constexpr double kLog2 = 0.693147180559945309417232121458176568;

std::vector<double> normalizeKernel(std::vector<double> kernel) {
    double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    if (sum <= 0.0) {
        throw std::runtime_error("Kernel normalization failed because the sum was non-positive.");
    }
    for (double &v : kernel) {
        v /= sum;
    }
    return kernel;
}

} // namespace

std::vector<double> buildGaussian1DKernel(double sigma_vox, int half_width) {
    if (sigma_vox <= 0.0) {
        throw std::runtime_error("Gaussian sigma must be positive.");
    }
    const int width = 2 * half_width + 1;
    std::vector<double> kernel(static_cast<std::size_t>(width), 0.0);
    for (int i = -half_width; i <= half_width; ++i) {
        const double x = static_cast<double>(i);
        const double value = std::exp(-(x * x) / (2.0 * sigma_vox * sigma_vox));
        kernel[static_cast<std::size_t>(i + half_width)] = value;
    }
    return normalizeKernel(std::move(kernel));
}

SeparableKernel3D buildGaussianPSF(double fwhm_x_mm,
                                   double fwhm_y_mm,
                                   double fwhm_z_mm,
                                   double voxsize_x_mm,
                                   double voxsize_y_mm,
                                   double voxsize_z_mm,
                                   int half_width) {
    if (voxsize_x_mm <= 0.0 || voxsize_y_mm <= 0.0 || voxsize_z_mm <= 0.0) {
        throw std::runtime_error("Voxel sizes must be positive when building the PSF.");
    }

    const double fwhm_x_vox = fwhm_x_mm / voxsize_x_mm;
    const double fwhm_y_vox = fwhm_y_mm / voxsize_y_mm;
    const double fwhm_z_vox = fwhm_z_mm / voxsize_z_mm;

    const double sigma_x = fwhm_x_vox / (2.0 * std::sqrt(2.0 * kLog2));
    const double sigma_y = fwhm_y_vox / (2.0 * std::sqrt(2.0 * kLog2));
    const double sigma_z = fwhm_z_vox / (2.0 * std::sqrt(2.0 * kLog2));

    SeparableKernel3D out;
    out.kx = buildGaussian1DKernel(sigma_x, half_width);
    out.ky = buildGaussian1DKernel(sigma_y, half_width);
    out.kz = buildGaussian1DKernel(sigma_z, half_width);

    const std::size_t nx = out.kx.size();
    const std::size_t ny = out.ky.size();
    const std::size_t nz = out.kz.size();
    out.kernel3d = Volume3D(nx, ny, nz, 0.0);

    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                out.kernel3d(x, y, z) = out.kx[x] * out.ky[y] * out.kz[z];
            }
        }
    }
    return out;
}

} // namespace pvc
