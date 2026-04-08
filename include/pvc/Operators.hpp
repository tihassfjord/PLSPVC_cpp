#pragma once

#include <vector>

#include "pvc/PSF.hpp"
#include "pvc/Volume.hpp"

namespace pvc {

struct ObjectiveResult {
    double value = 0.0;
    Volume3D gradient;
};

VectorField3D gradient3D(const Volume3D &input, const std::string &mode);
VectorField3D applyPLSOperator(const VectorField3D &guidance_gradient, const VectorField3D &input);
VectorField3D shrinkPLS(const VectorField3D &b_gu, const VectorField3D &bregman, double lambda);
VectorField3D updateBregman(const VectorField3D &b_gu, const VectorField3D &d, const VectorField3D &bregman);
Volume3D sumDiagonalBackwardDivergence(const VectorField3D &field);

Volume3D convolveSame(const Volume3D &input, const Volume3D &kernel);
Volume3D separableConvolveSame(const Volume3D &input,
                               const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz);
Volume3D flipKernel(const Volume3D &kernel);

Volume3D gradientMagnitudeImage(const Volume3D &input);
Volume3D normalizeToUnitMax(const Volume3D &input);

} // namespace pvc
