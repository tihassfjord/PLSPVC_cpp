#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "pvc/NiftiIO.hpp"
#include "pvc/PSF.hpp"
#include "pvc/SplitBregmanPVC.hpp"

int main() {
    try {
        constexpr std::size_t nx = 5;
        constexpr std::size_t ny = 5;
        constexpr std::size_t nz = 5;
        constexpr std::size_t nt = 2;

        pvc::NiftiImage pet4d;
        pet4d.dims = {4, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), static_cast<int>(nt), 1, 1, 1};
        pet4d.pixdim = {1.0, 1.5, 1.5, 2.0, 30.0, 1.0, 1.0, 1.0};
        pet4d.datatype = 64;
        pet4d.data.resize(nx * ny * nz * nt, 0.0);

        for (std::size_t t = 0; t < nt; ++t) {
            for (std::size_t z = 0; z < nz; ++z) {
                for (std::size_t y = 0; y < ny; ++y) {
                    for (std::size_t x = 0; x < nx; ++x) {
                        const double dx = static_cast<double>(x) - 2.0;
                        const double dy = static_cast<double>(y) - 2.0;
                        const double dz = static_cast<double>(z) - 2.0;
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        const double base = (r2 <= 2.0) ? 1.0 : 0.1;
                        pet4d.data[pet4d.index(x, y, z, t)] = base * (1.0 + 0.2 * static_cast<double>(t));
                    }
                }
            }
        }

        const pvc::Volume3D shared_guidance = pet4d.meanOverTime();
        const pvc::SeparableKernel3D psf = pvc::buildGaussianPSF(1.6, 1.6, 1.8, 1.5, 1.5, 2.0);
        pvc::SolverOptions options;
        options.niter = 1;
        options.verbose = false;

        std::vector<pvc::Volume3D> corrected_frames;
        corrected_frames.reserve(nt);
        for (std::size_t t = 0; t < nt; ++t) {
            const pvc::Volume3D frame = pet4d.frame(t);
            const pvc::SolverResult result = pvc::runSplitBregmanPVC(frame, shared_guidance, psf, options);
            corrected_frames.push_back(result.corrected);
        }

        const std::filesystem::path out_path = std::filesystem::temp_directory_path() / "pls_pvc_test_4d.nii.gz";
        pvc::writeNifti(out_path.string(), pet4d, corrected_frames);
        const pvc::NiftiImage reloaded = pvc::readNifti(out_path.string());

        if (!reloaded.is4D()) {
            throw std::runtime_error("Reloaded output did not keep a 4D shape.");
        }
        if (reloaded.nt() != nt) {
            throw std::runtime_error("Reloaded output has the wrong number of time frames.");
        }
        if (reloaded.voxelCountTotal() != pet4d.voxelCountTotal()) {
            throw std::runtime_error("Reloaded output has the wrong voxel count.");
        }
        for (double v : reloaded.data) {
            if (!std::isfinite(v)) {
                throw std::runtime_error("Reloaded 4D output contains a non-finite value.");
            }
        }

        std::filesystem::remove(out_path);
        std::cout << "4D smoke test passed.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "4D smoke test failed: " << e.what() << '\n';
        return 1;
    }
}
