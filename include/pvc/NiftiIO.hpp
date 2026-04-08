#pragma once

#include <string>
#include <vector>

#include "pvc/Volume.hpp"

namespace pvc {

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
