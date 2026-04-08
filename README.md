# PLS PVC C++

A file-based C++ port of the 3D PLS-based PET partial-volume correction method, organized so the **mathematics stays recognizable** while the implementation is no longer tied to MATLAB.

## What is preserved from the reference method

The solver keeps the same high-level structure as the reference MATLAB implementation:

- PET is rescaled by its maximum value before optimization, then scaled back afterwards.
- The guidance image is normalized before the guidance gradient field is formed.
- The PLS field uses normalized guidance gradients with a small stabilizer (`mr_smooth`).
- The algorithm alternates between:
  1. solving the `u` subproblem with one-step gradient descent plus line search,
  2. solving the `d` subproblem with soft thresholding,
  3. updating the Bregman variable `b`,
  4. adapting `lambda` with residual balancing.
- The stopping logic follows the same idea as the reference code: detect stalled progress, then switch to fixed `lambda`, then stop when the normalized primal residual is small enough.
- The finite-difference boundary handling follows the mirrored boundary behaviour in the MATLAB code, rather than silently switching to a different stencil convention.

So the core method is not replaced by a “kind of similar” method. The **reference path** is still the reference path.

## What is intentionally improved in the implementation

This is not a line-for-line MATLAB clone. Some things were changed on purpose so the C++ version is a better base for future acceleration:

- contiguous 3D volumes instead of MATLAB cell arrays and repeated temporary allocations,
- explicit 3D scalar and vector-field operators,
- separable Gaussian PSF convolution for the scanner blur term,
- optional OpenMP parallelism on CPU,
- minimal built-in NIfTI read/write so you can run it directly on `.nii` and `.nii.gz`,
- a guidance fallback system for cases where no MR is available,
- frame-by-frame execution for **4D PET** with 4D NIfTI output.
- **EXPERIMENTAL**: Initial CUDA GPU support for accelerating iterative computations.

That fallback system is an extension, not part of the original paper code.

## Build

### CPU-Only (Default)
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

### With CUDA (GPU Backend)
```bash
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build -j
ctest --test-dir build
```

Dependencies:

- C++17 compiler
- CMake >= 3.16
- zlib
- optionally OpenMP
- optionally CUDA Toolkit (>= 11.0)

## Command-line usage

### 3D PET with MR guidance

```bash
./build/pls_pvc \
  --pet pet_frame.nii.gz \
  --mr mr.nii.gz \
  --out pet_pvc.nii.gz \
  --fwhm-x 1.6 \
  --fwhm-y 1.6 \
  --fwhm-z 1.8 \
  --guidance-mode mr \
  --niter 100 \
  --mu 17
```

### 3D PET without MR

```bash
./build/pls_pvc \
  --pet pet_frame.nii.gz \
  --out pet_pvc.nii.gz \
  --fwhm-x 1.6 \
  --fwhm-y 1.6 \
  --fwhm-z 1.8 \
  --guidance-mode pet-smoothed \
  --smooth-sigma 1.0
```

### 4D PET without MR, using the PET time average as structural surrogate

```bash
./build/pls_pvc \
  --pet dynamic_pet.nii.gz \
  --out dynamic_pet_pvc.nii.gz \
  --fwhm-x 1.6 \
  --fwhm-y 1.6 \
  --fwhm-z 1.8 \
  --guidance-mode average-4d
```

If `--guidance-mode` is omitted and the PET input is 4D, the code now defaults to `average-4d` because that is usually the least silly PET-only fallback.

### 4D PET using another 4D scan as surrogate guidance source

```bash
./build/pls_pvc \
  --pet dynamic_pet.nii.gz \
  --out dynamic_pet_pvc.nii.gz \
  --fwhm-x 1.6 \
  --fwhm-y 1.6 \
  --fwhm-z 1.8 \
  --guidance-mode average-4d \
  --guidance dynamic_surrogate.nii.gz
```

## 4D PET handling

The corrected PET input can now be either:

- **3D**: one corrected 3D output volume
- **4D**: each time frame is corrected independently, then written back as one corrected 4D NIfTI series

Important practical detail:

- the **optimization itself is still 3D per frame**
- this is not yet a coupled spatio-temporal inverse solver
- it is the right first implementation because it preserves the reference mathematics while making dynamic PET usable immediately

### Guidance behaviour for 4D PET

- `mr`: same 3D MR reused for every frame
- `average-4d`: one shared 3D guidance image built by averaging a 4D series over time
- `pet`, `pet-smoothed`, `pet-edge`: guidance built separately from each PET frame
- `external`: one shared external 3D guidance image reused for every frame

For noisy dynamic PET, `average-4d` is usually the best PET-only starting point because it produces a more stable structural surrogate than per-frame guidance.

## Parameters and how to think about them

### Scanner blur parameters

`--fwhm-x`, `--fwhm-y`, `--fwhm-z`

These describe the scanner point-spread function in millimetres. They matter a lot. If these are wrong, the PVC will still run, but it will be correcting the wrong blur model. That is a polished way of being wrong.

Use measured PSF values when possible.

### Iteration count

`--niter`

Maximum number of outer Split-Bregman iterations. Start with the default. Increase only if the per-iteration change and residuals suggest the method is still moving meaningfully.

### Data-fidelity weight

`--mu`

This controls how strongly the corrected image is forced to stay consistent with the blurred observed PET data.

- larger `mu` = more trust in the measured PET data term,
- smaller `mu` = more freedom for the regularized solution to move.

Too large can leave you under-corrected. Too small can let noise and structure hallucination creep in.

### Split-Bregman penalty parameter

`--lambda-init`

This is mainly a **numerical splitting parameter**, not the same thing as the regularization strength in a simplistic “bigger is smoother” sense.

In ideal operator-splitting land it mostly affects convergence behaviour, not the final solution. In real code it still matters. If convergence is erratic, this is one of the first things worth adjusting.

### Guidance gradient stabilizer

`--mr-smooth`

A small positive number added inside the normalization of the guidance gradient:

```text
sqrt(gx^2 + gy^2 + gz^2 + mr_smooth^2)
```

It prevents division by zero and keeps flat regions from blowing up numerically.

Leave it alone unless you have a very specific reason.

### Relative update tolerance

`--tol`

If the relative change in the solution is smaller than this for several iterations in a row, the solver treats progress as stalled and changes how `lambda` is handled.

### Stagnation counter

`--stop-count`

How many consecutive small updates are required before the solver decides that adaptive `lambda` has done its job and it is time to move to the fixed-`lambda` phase.

### Residual threshold scaling

`--epsilon-scale`

The reference MATLAB code normalizes the stopping threshold to a specific phantom size. This parameter preserves that logic, but exposes the scaling so you can control how strict the residual stopping rule should be.

### PET-only guidance controls

`--smooth-sigma`

Used in PET-based fallback modes to smooth the PET before forming guidance. Larger values suppress noisy gradients more strongly, but if you push too far you also blur away the structure you hoped to preserve.

`--edge-weight`

Used in `pet-edge`. This controls how much the PET-derived edge magnitude contributes to the surrogate guidance image. It is a heuristic knob, not sacred scripture.

## Guidance modes

### `mr`
Use the MR volume as structural guidance. This is the closest to the intended original use.

### `pet`
Use the PET image itself as guidance. This is the simplest fallback when no MR exists, but it risks reinforcing PET blur structure instead of introducing genuinely sharper anatomical information.

### `pet-smoothed`
Use a smoothed PET image as guidance. This is the recommended no-MR starting point for a single static frame because it reduces noisy PET gradients while still giving the PLS term structural hints.

### `pet-edge`
Use a smoothed PET image plus a weighted edge-magnitude image derived from PET. This is a heuristic extension for cases where PET edges are informative enough to help.

This is useful to try, but it is **not** part of the original reference method.

### `external`
Use an external 3D NIfTI volume as the guidance image.

### `average-4d`
Average a 4D NIfTI across time and use that 3D average as the guidance image. If `--guidance` is omitted, the program will average the **input PET time series itself**. This is the practical route when you do not have MR but do have a dynamic PET dataset and want a more stable structural surrogate.

## Current scope and limitations

- The corrected PET input can now be **3D or 4D**.
- A 4D PET is corrected **frame by frame**, not with a coupled temporal prior.
- MR guidance is currently expected to be **3D**.
- External `average-4d` guidance requires a **4D** NIfTI.
- NIfTI support is intentionally minimal and focuses on common scalar datatypes in single-file `.nii` / `.nii.gz` format.
- This version is a **CPU reference implementation with a future-friendly layout**. It is already more structured than the MATLAB code, but it is not yet the final word in performance.

## File layout

- `include/pvc/Volume.hpp` — 3D scalar/vector field containers and NIfTI image metadata wrapper
- `include/pvc/NiftiIO.hpp` + `src/NiftiIO.cpp` — minimal NIfTI I/O, now including 4D write support
- `include/pvc/PSF.hpp` + `src/PSF.cpp` — Gaussian PSF builder matching the reference cropped kernel logic
- `include/pvc/Operators.hpp` + `src/Operators.cpp` — gradients, PLS operator, shrinkage, convolution
- `include/pvc/Guidance.hpp` + `src/Guidance.cpp` — MR and PET-based guidance construction
- `include/pvc/SplitBregmanPVC.hpp` + `src/SplitBregmanPVC.cpp` — main solver
- `src/main.cpp` — CLI, including the 4D frame loop
- `tests/smoke_test.cpp` — small synthetic 3D smoke test
- `tests/smoke_test_4d.cpp` — small synthetic 4D smoke test

## What should happen next

If you want this to become the **fast** version rather than just the **clean reference** version, the next sensible steps are:

1. profile the CPU code on real PET sizes,
2. replace the remaining heavy passes with a backend abstraction,
3. add a CUDA backend for the blur operator and vector-field passes,
4. validate numerically against the MATLAB implementation on the same test case.

That order is much less glamorous than jumping straight to CUDA, but it is also much less stupid.
