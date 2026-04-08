#include <cmath>
#include <iostream>
#include <stdexcept>

#include "pvc/Guidance.hpp"
#include "pvc/PSF.hpp"
#include "pvc/SplitBregmanPVC.hpp"

int main() {
    try {
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
                    mr(x, y, z) = (r2 < 9.0) ? 1.0 : 0.1;
                    pet(x, y, z) = (r2 < 9.0) ? 0.8 : 0.05;
                }
            }
        }

        const pvc::SeparableKernel3D psf = pvc::buildGaussianPSF(1.6, 1.6, 1.6, 1.0, 1.0, 1.0);
        pvc::SolverOptions options;
        options.niter = 1;
        options.verbose = false;

        const pvc::SolverResult result = pvc::runSplitBregmanPVC(pet, mr, psf, options);
        if (result.corrected.size() != pet.size()) {
            throw std::runtime_error("Corrected volume has the wrong size.");
        }
        for (double v : result.corrected.data()) {
            if (!std::isfinite(v)) {
                throw std::runtime_error("Corrected volume contains a non-finite value.");
            }
        }

        std::cout << "Smoke test passed.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Smoke test failed: " << e.what() << '\n';
        return 1;
    }
}
