#pragma once

#include <string>

#include "pvc/Volume.hpp"

namespace pvc {

enum class GuidanceMode {
    Mr,
    Pet,
    PetSmoothed,
    PetEdgeEnhanced,
    External3D,
    Average4D
};

struct GuidanceOptions {
    GuidanceMode mode = GuidanceMode::Mr;
    std::string external_guidance_path;
    double smooth_sigma_vox = 1.0;
    double edge_weight = 0.35;
};

GuidanceMode parseGuidanceMode(const std::string &value);
std::string guidanceModeToString(GuidanceMode mode);

// Builds the structural guidance image that feeds the PLS gradient field.
// Important: when MR is provided and mode=Mr, this preserves the original intended use of the method.
// PET-based modes are practical fallback extensions for cases where no MR is available.
Volume3D buildGuidanceImage(const Volume3D &pet,
                            const Volume3D *mr,
                            const GuidanceOptions &options);

} // namespace pvc
