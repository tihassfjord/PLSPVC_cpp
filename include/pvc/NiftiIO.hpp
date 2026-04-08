#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>

#include "pvc/Volume.hpp"

namespace pvc {

struct NiftiImage {
    std::vector<int> dims;
    std::vector<double> pixdim;
    int datatype = 0;
    bool little_endian = true;
    std::array<std::uint8_t, 348> original_header{};

    std::vector<double> data;

    bool is3D() const {
        return dims.size() >= 4 && dims[1] > 1 && dims[2] > 1 && dims[3] > 1;
    }
    
    std::size_t voxelCount3D() const {
        return is3D() ? static_cast<std::size_t>(dims[1] * dims[2] * dims[3]) : 0;
    }

    std::size_t voxelCountTotal() const {
        std::size_t total = 1;
        for (std::size_t i = 1; i < dims.size() && dims[i] > 0; ++i) {
            total *= static_cast<std::size_t>(dims[i]);
        }
        return total;
    }
};

// Minimal NIfTI-1 reader/writer for single-file .nii and .nii.gz.
// This is intentionally small and dependency-light so the PVC codebase is easy to build.
// It supports common scalar datatypes and converts everything to double on read.
NiftiImage readNifti(const std::string &path);

// Write a single corrected 3D frame.
void writeNifti(const std::string &path, const NiftiImage &reference_header, const Volume3D &volume);

// Write a full corrected dynamic series as a 4D NIfTI.
// The reference header is used to preserve voxel spacing and time-axis metadata when available.
void writeNifti(const std::string &path, const NiftiImage &reference_header, const std::vector<Volume3D> &volumes);

} // namespace pvc
