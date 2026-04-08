#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pvc/Guidance.hpp"
#include "pvc/NiftiIO.hpp"
#include "pvc/PSF.hpp"
#include "pvc/SplitBregmanPVC.hpp"
#if PVC_USE_CUDA
#include "pvc/CudaSplitBregmanPVC.hpp"
#endif

namespace {

using ArgMap = std::unordered_map<std::string, std::string>;

void printHelp() {
    std::cout << R"(PLS PVC C++ reference implementation

Required arguments:
  --pet PATH                 Input PET NIfTI (.nii or .nii.gz). Can be 3D or 4D.
  --out PATH                 Output corrected NIfTI (.nii or .nii.gz)
  --fwhm-x MM                System FWHM in x (mm)
  --fwhm-y MM                System FWHM in y (mm)
  --fwhm-z MM                System FWHM in z (mm)

Optional arguments:
  --mr PATH                  Structural MR NIfTI (3D). Reused for all PET frames.
  --guidance-mode MODE       mr | pet | pet-smoothed | pet-edge | external | average-4d
  --guidance PATH            External 3D guidance NIfTI for mode=external, or 4D NIfTI for mode=average-4d.
                             For mode=average-4d, omitting --guidance means: use the input PET time average itself.
  --niter N                  Maximum number of Split-Bregman outer iterations (default: 100)
  --mu VALUE                 Data-fidelity weight. Larger = trust blurred PET data more strongly (default: 17)
  --lambda-init VALUE        Initial Split-Bregman penalty parameter. Affects convergence behaviour more than the target solution (default: 68)
  --eta VALUE                Running average parameter used in the line-search majorizer (default: 0.995)
  --mr-smooth VALUE          Small stabilizer added inside ||grad(g)|| before guidance normalization (default: 6e-4)
  --tol VALUE                Relative update tolerance for declaring stalled progress (default: 1e-3)
  --stop-count N             Number of consecutive small updates before switching lambda handling (default: 3)
  --epsilon-scale VALUE      MATLAB-normalized residual threshold scaling (default: 0.1)
  --smooth-sigma VALUE       Gaussian smoothing sigma in voxels for PET-based fallback guidance (default: 1.0)
  --edge-weight VALUE        Weight on edge magnitude for pet-edge guidance mode (default: 0.35)
  --gpu                      Use CUDA GPU acceleration (requires CUDA build)
  --quiet                    Suppress per-iteration logging
  --help                     Show this message

Practical guidance:
  * If you have MR, use --guidance-mode mr and pass --mr.
  * If you do not have MR and the PET is dynamic, average-4d is usually the most sensible PET-only structural surrogate.
  * If you do not have MR and only have a noisy static frame, start with --guidance-mode pet-smoothed.
  * Use pet-edge only if the PET has enough contrast to produce meaningful edges.
  * FWHM values should come from measured scanner resolution, not guesswork dressed as optimism.
)";
}

ArgMap parseArgs(int argc, char **argv) {
    ArgMap args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h" || key == "--quiet" || key == "--gpu") {
            args[key] = "1";
            continue;
        }
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected positional argument: " + key);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value after argument: " + key);
        }
        args[key] = argv[++i];
    }
    return args;
}

double getDouble(const ArgMap &args, const std::string &key, double default_value, bool required = false) {
    const auto it = args.find(key);
    if (it == args.end()) {
        if (required) {
            throw std::runtime_error("Missing required argument: " + key);
        }
        return default_value;
    }
    return std::stod(it->second);
}

int getInt(const ArgMap &args, const std::string &key, int default_value, bool required = false) {
    const auto it = args.find(key);
    if (it == args.end()) {
        if (required) {
            throw std::runtime_error("Missing required argument: " + key);
        }
        return default_value;
    }
    return std::stoi(it->second);
}

std::string getString(const ArgMap &args, const std::string &key, const std::string &default_value = {}, bool required = false) {
    const auto it = args.find(key);
    if (it == args.end()) {
        if (required) {
            throw std::runtime_error("Missing required argument: " + key);
        }
        return default_value;
    }
    return it->second;
}

pvc::Volume3D resolveStaticGuidanceFromMode(const pvc::NiftiImage &pet_nifti,
                                            const pvc::Volume3D &first_pet_frame,
                                            const std::optional<pvc::Volume3D> &mr,
                                            const pvc::GuidanceOptions &guidance_options) {
    if (guidance_options.mode == pvc::GuidanceMode::External3D) {
        if (guidance_options.external_guidance_path.empty()) {
            throw std::runtime_error("Guidance mode 'external' requires --guidance PATH.");
        }
        const pvc::NiftiImage guidance_nifti = pvc::readNifti(guidance_options.external_guidance_path);
        if (guidance_nifti.nt() != 1) {
            throw std::runtime_error("Guidance mode 'external' requires a 3D NIfTI.");
        }
        pvc::Volume3D guidance = guidance_nifti.frame(0);
        first_pet_frame.requireSameShape(guidance, "PET vs external guidance");
        return guidance;
    }

    if (guidance_options.mode == pvc::GuidanceMode::Average4D) {
        pvc::NiftiImage guidance_nifti;
        if (!guidance_options.external_guidance_path.empty()) {
            guidance_nifti = pvc::readNifti(guidance_options.external_guidance_path);
            if (!guidance_nifti.is4D()) {
                throw std::runtime_error("Guidance mode 'average-4d' with --guidance requires a 4D NIfTI.");
            }
        } else {
            if (!pet_nifti.is4D()) {
                throw std::runtime_error("Guidance mode 'average-4d' without --guidance only makes sense when the main PET input is 4D.");
            }
            guidance_nifti = pet_nifti;
        }
        pvc::Volume3D guidance = guidance_nifti.meanOverTime();
        first_pet_frame.requireSameShape(guidance, "PET vs averaged 4D guidance");
        return guidance;
    }

    const pvc::Volume3D *mr_ptr = mr ? &(*mr) : nullptr;
    return pvc::buildGuidanceImage(first_pet_frame, mr_ptr, guidance_options);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const ArgMap args = parseArgs(argc, argv);
        if (args.count("--help") > 0 || argc == 1) {
            printHelp();
            return 0;
        }

        const std::string pet_path = getString(args, "--pet", {}, true);
        const std::string out_path = getString(args, "--out", {}, true);
        const std::string mr_path = getString(args, "--mr");
        const double fwhm_x = getDouble(args, "--fwhm-x", 0.0, true);
        const double fwhm_y = getDouble(args, "--fwhm-y", 0.0, true);
        const double fwhm_z = getDouble(args, "--fwhm-z", 0.0, true);

        pvc::SolverOptions solver_options;
        solver_options.niter = getInt(args, "--niter", solver_options.niter);
        solver_options.mu = getDouble(args, "--mu", solver_options.mu);
        solver_options.lambda_init = getDouble(args, "--lambda-init", solver_options.lambda_init);
        solver_options.eta = getDouble(args, "--eta", solver_options.eta);
        solver_options.mr_smooth = getDouble(args, "--mr-smooth", solver_options.mr_smooth);
        solver_options.convergence_tol = getDouble(args, "--tol", solver_options.convergence_tol);
        solver_options.stop_counter_limit = getInt(args, "--stop-count", solver_options.stop_counter_limit);
        solver_options.epsilon_scale = getDouble(args, "--epsilon-scale", solver_options.epsilon_scale);
        solver_options.verbose = (args.count("--quiet") == 0);

        pvc::GuidanceOptions guidance_options;
        guidance_options.smooth_sigma_vox = getDouble(args, "--smooth-sigma", guidance_options.smooth_sigma_vox);
        guidance_options.edge_weight = getDouble(args, "--edge-weight", guidance_options.edge_weight);
        guidance_options.external_guidance_path = getString(args, "--guidance");
        if (args.count("--guidance-mode") > 0) {
            guidance_options.mode = pvc::parseGuidanceMode(getString(args, "--guidance-mode"));
        }

        const pvc::NiftiImage pet_nifti = pvc::readNifti(pet_path);
        if (!pet_nifti.is3D()) {
            throw std::runtime_error("Input PET NIfTI must be at least 3D.");
        }
        const std::size_t nframes = pet_nifti.nt();
        const pvc::Volume3D first_pet_frame = pet_nifti.frame(0);

        if (args.count("--guidance-mode") == 0) {
            if (!mr_path.empty()) {
                guidance_options.mode = pvc::GuidanceMode::Mr;
            } else if (nframes > 1) {
                guidance_options.mode = pvc::GuidanceMode::Average4D;
            } else {
                guidance_options.mode = pvc::GuidanceMode::PetSmoothed;
            }
        }

        std::optional<pvc::Volume3D> mr;
        if (!mr_path.empty()) {
            const pvc::NiftiImage mr_nifti = pvc::readNifti(mr_path);
            if (mr_nifti.nt() != 1) {
                throw std::runtime_error("MR guidance must be 3D. One MR reused across frames is fine; a 4D MR time series is not supported here.");
            }
            mr = mr_nifti.frame(0);
            first_pet_frame.requireSameShape(*mr, "PET vs MR");
        }

        const double voxsize_x = (pet_nifti.pixdim.size() > 1 && pet_nifti.pixdim[1] > 0.0) ? pet_nifti.pixdim[1] : 1.0;
        const double voxsize_y = (pet_nifti.pixdim.size() > 2 && pet_nifti.pixdim[2] > 0.0) ? pet_nifti.pixdim[2] : 1.0;
        const double voxsize_z = (pet_nifti.pixdim.size() > 3 && pet_nifti.pixdim[3] > 0.0) ? pet_nifti.pixdim[3] : 1.0;
        const pvc::SeparableKernel3D psf = pvc::buildGaussianPSF(fwhm_x, fwhm_y, fwhm_z,
                                                                 voxsize_x, voxsize_y, voxsize_z);

        if (solver_options.verbose) {
            std::cout << "Guidance mode: " << pvc::guidanceModeToString(guidance_options.mode) << '\n';
            std::cout << "PET voxel size (mm): [" << voxsize_x << ", " << voxsize_y << ", " << voxsize_z << "]\n";
            if (nframes > 1) {
                std::cout << "Dynamic PET detected: correcting " << nframes << " frame(s) independently and writing a 4D output series.\n";
            }
        }

        const bool static_guidance_mode = guidance_options.mode == pvc::GuidanceMode::Mr ||
                                          guidance_options.mode == pvc::GuidanceMode::External3D ||
                                          guidance_options.mode == pvc::GuidanceMode::Average4D;

        std::optional<pvc::Volume3D> shared_guidance;
        if (static_guidance_mode) {
            shared_guidance = resolveStaticGuidanceFromMode(pet_nifti, first_pet_frame, mr, guidance_options);
        }

        const bool use_gpu = args.count("--gpu") > 0;
#if !PVC_USE_CUDA
        if (use_gpu) {
            throw std::runtime_error("--gpu requested but this binary was built without CUDA support. Rebuild with -DUSE_CUDA=ON.");
        }
#endif

        std::vector<pvc::Volume3D> corrected_frames;
        corrected_frames.reserve(nframes);
        pvc::SolverSummary last_summary{};

        for (std::size_t t = 0; t < nframes; ++t) {
            const pvc::Volume3D pet_frame = pet_nifti.frame(t);
            pvc::Volume3D guidance_frame;
            if (shared_guidance) {
                guidance_frame = *shared_guidance;
            } else {
                const pvc::Volume3D *mr_ptr = mr ? &(*mr) : nullptr;
                guidance_frame = pvc::buildGuidanceImage(pet_frame, mr_ptr, guidance_options);
            }
            pet_frame.requireSameShape(guidance_frame, "PET vs guidance frame");

            if (solver_options.verbose && nframes > 1) {
                std::cout << "\n=== Solving frame " << (t + 1) << "/" << nframes << " ===\n";
            }

            pvc::SolverResult result;
#if PVC_USE_CUDA
            if (use_gpu) {
                result = pvc::runSplitBregmanPVC_CUDA(pet_frame, guidance_frame, psf, solver_options);
            } else
#endif
            {
                result = pvc::runSplitBregmanPVC(pet_frame, guidance_frame, psf, solver_options);
            }
            corrected_frames.push_back(result.corrected);
            last_summary = result.summary;
        }

        if (nframes == 1) {
            pvc::writeNifti(out_path, pet_nifti, corrected_frames.front());
        } else {
            pvc::writeNifti(out_path, pet_nifti, corrected_frames);
        }

        std::cout << "Finished " << nframes << " frame(s).\n"
                  << "Last-frame lambda = " << last_summary.final_lambda << '\n'
                  << "Last-frame normalized primal residual = " << last_summary.final_primal_residual << '\n'
                  << "Last-frame normalized dual residual = " << last_summary.final_dual_residual << '\n'
                  << "Output written to: " << out_path << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
