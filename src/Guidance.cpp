#include "pvc/Guidance.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "pvc/Operators.hpp"
#include "pvc/PSF.hpp"

namespace pvc {
namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

Volume3D gaussianSmooth(const Volume3D &input, double sigma_vox) {
    if (sigma_vox <= 0.0) {
        return input;
    }
    const int half_width = std::max(1, static_cast<int>(std::ceil(3.0 * sigma_vox)));
    const std::vector<double> k = buildGaussian1DKernel(sigma_vox, half_width);
    return separableConvolveSame(input, k, k, k);
}

} // namespace

GuidanceMode parseGuidanceMode(const std::string &value) {
    const std::string key = toLower(value);
    if (key == "mr") return GuidanceMode::Mr;
    if (key == "pet") return GuidanceMode::Pet;
    if (key == "pet-smoothed" || key == "pet_smoothed") return GuidanceMode::PetSmoothed;
    if (key == "pet-edge" || key == "pet_edge" || key == "pet-edge-enhanced") return GuidanceMode::PetEdgeEnhanced;
    if (key == "external") return GuidanceMode::External3D;
    if (key == "average-4d" || key == "average_4d") return GuidanceMode::Average4D;
    throw std::runtime_error("Unknown guidance mode: " + value);
}

std::string guidanceModeToString(GuidanceMode mode) {
    switch (mode) {
    case GuidanceMode::Mr: return "mr";
    case GuidanceMode::Pet: return "pet";
    case GuidanceMode::PetSmoothed: return "pet-smoothed";
    case GuidanceMode::PetEdgeEnhanced: return "pet-edge";
    case GuidanceMode::External3D: return "external";
    case GuidanceMode::Average4D: return "average-4d";
    }
    return "unknown";
}

Volume3D buildGuidanceImage(const Volume3D &pet,
                            const Volume3D *mr,
                            const GuidanceOptions &options) {
    switch (options.mode) {
    case GuidanceMode::Mr:
        if (!mr) {
            throw std::runtime_error("Guidance mode 'mr' was requested, but no MR image was provided.");
        }
        return normalizeToUnitMax(*mr);

    case GuidanceMode::Pet:
        return normalizeToUnitMax(pet);

    case GuidanceMode::PetSmoothed: {
        const Volume3D smooth = gaussianSmooth(normalizeToUnitMax(pet), options.smooth_sigma_vox);
        return normalizeToUnitMax(smooth);
    }

    case GuidanceMode::PetEdgeEnhanced: {
        const Volume3D smooth = gaussianSmooth(normalizeToUnitMax(pet), options.smooth_sigma_vox);
        const Volume3D edge = normalizeToUnitMax(gradientMagnitudeImage(smooth));
        Volume3D guidance = smooth;
        guidance.addScaledInPlace(edge, options.edge_weight);
        return normalizeToUnitMax(guidance);
    }

    case GuidanceMode::External3D:
    case GuidanceMode::Average4D:
        throw std::runtime_error("External guidance modes must be constructed from a NIfTI file before calling buildGuidanceImage.");
    }

    throw std::runtime_error("Unhandled guidance mode.");
}

} // namespace pvc
