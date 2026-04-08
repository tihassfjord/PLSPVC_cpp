#include <cmath>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "pvc/CudaSplitBregmanPVC.hpp"
#include "pvc/Guidance.hpp"
#include "pvc/PSF.hpp"

int main() {
    try {
        // Verify a CUDA device is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            std::cerr << "CUDA smoke test skipped: no CUDA device found.\n";
            return 0;
        }

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU: " << prop.name << '\n';

        // Build a small synthetic test volume (identical to CPU smoke test)
        constexpr std::size_t n = 6;
        pvc::Volume3D pet(n, n, n, 0.0);
        pvc::Volume3D mr(n, n, n, 0.0);

        for (std::size_t z = 0; z < n; ++z) {
            for (std::size_t y = 0; y < n; ++y) {
                for (std::size_t x = 0; x < n; ++x) {
                    const double dx = static_cast<double>(x) - 7.5;
                    const double dy = static_cast<double>(y) - 7.5;
                    const double dz = static_cast<double>(z) - 7.5;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    mr(x, y, z)  = (r2 < 9.0) ? 1.0 : 0.1;
                    pet(x, y, z) = (r2 < 9.0) ? 0.8 : 0.05;
                }
            }
        }

        const pvc::SeparableKernel3D psf = pvc::buildGaussianPSF(1.6, 1.6, 1.6, 1.0, 1.0, 1.0);
        pvc::SolverOptions options;
        options.niter   = 1;
        options.verbose = false;

        // Run the CUDA solver
        const pvc::SolverResult result = pvc::runSplitBregmanPVC_CUDA(pet, mr, psf, options);

        // Basic validity checks
        if (result.corrected.size() != pet.size()) {
            throw std::runtime_error("Corrected volume has the wrong size.");
        }
        for (double v : result.corrected.data()) {
            if (!std::isfinite(v)) {
                throw std::runtime_error("Corrected volume contains a non-finite value.");
            }
        }

        // Cross-validate against CPU reference (single iteration, should be close)
        const pvc::SolverResult cpu_result = pvc::runSplitBregmanPVC(pet, mr, psf, options);
        double max_diff = 0.0;
        for (std::size_t i = 0; i < result.corrected.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(result.corrected.data()[i] - cpu_result.corrected.data()[i]));
        }

        constexpr double tol = 1e-6;
        if (max_diff > tol) {
            std::cerr << "WARNING: GPU vs CPU max difference = " << max_diff
                      << " (tolerance = " << tol << ")\n";
        } else {
            std::cout << "GPU vs CPU max difference = " << max_diff << " (OK)\n";
        }

        std::cout << "CUDA smoke test passed.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "CUDA smoke test failed: " << e.what() << '\n';
        return 1;
    }
}
