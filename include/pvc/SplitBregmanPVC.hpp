#pragma once

#include <string>

#include "pvc/PSF.hpp"
#include "pvc/Volume.hpp"

namespace pvc {

struct SolverOptions {
    int niter = 100;
    double mu = 17.0;
    double lambda_init = 68.0;
    double eta = 0.995;
    double mr_smooth = 6e-4;
    double convergence_tol = 1e-3;
    int stop_counter_limit = 3;
    double epsilon_scale = 0.1;
    int line_search_max_backtracks = 8;
    double line_search_epsilon = 1e-4;
    double line_search_rho = 0.4;
    bool verbose = true;
};

struct SolverSummary {
    int iterations_run = 0;
    double final_lambda = 0.0;
    double final_primal_residual = 0.0;
    double final_dual_residual = 0.0;
};

struct SolverResult {
    Volume3D corrected;
    SolverSummary summary;
};

SolverResult runSplitBregmanPVC(const Volume3D &pet,
                                const Volume3D &guidance_image,
                                const SeparableKernel3D &psf,
                                const SolverOptions &options);

} // namespace pvc
