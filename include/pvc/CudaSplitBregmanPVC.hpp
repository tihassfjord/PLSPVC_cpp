#pragma once

#include "pvc/PSF.hpp"
#include "pvc/SplitBregmanPVC.hpp"
#include "pvc/Volume.hpp"

namespace pvc {

// GPU-accelerated Split-Bregman PVC solver.
// Same interface as the CPU version.  Guidance image must be pre-built on the
// host; the function copies data to the GPU, runs the entire iterative loop on
// device, and copies only the final corrected volume back.
SolverResult runSplitBregmanPVC_CUDA(const Volume3D &pet,
                                     const Volume3D &guidance_image,
                                     const SeparableKernel3D &psf,
                                     const SolverOptions &options);

} // namespace pvc
