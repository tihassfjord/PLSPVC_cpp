#pragma once

#include <vector>

#include "pvc/Volume.hpp"

namespace pvc {

struct SeparableKernel3D {
    std::vector<double> kx;
    std::vector<double> ky;
    std::vector<double> kz;
    Volume3D kernel3d;
};

// Builds the same 13x13x13 cropped Gaussian PSF that the reference MATLAB code uses,
// but stores it both as a full 3D kernel and as separable 1D kernels for faster CPU convolution.
SeparableKernel3D buildGaussianPSF(double fwhm_x_mm,
                                   double fwhm_y_mm,
                                   double fwhm_z_mm,
                                   double voxsize_x_mm,
                                   double voxsize_y_mm,
                                   double voxsize_z_mm,
                                   int half_width = 6);

std::vector<double> buildGaussian1DKernel(double sigma_vox, int half_width);

} // namespace pvc
