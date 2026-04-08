# CUDA Backend Implementation Summary

## New Files Created (6 files)

| File | Purpose |
|------|---------|
| `include/pvc/DeviceVolume3D.cuh` | RAII wrapper for `cudaMalloc`/`cudaFree` — `DeviceVolume3D` and `DeviceVectorField3D` structs with move semantics, host/device transfers |
| `include/pvc/CudaOperators.cuh` | Public API for all CUDA kernel operations |
| `src/CudaOperators.cu` | All GPU kernels (~450 lines) |
| `include/pvc/CudaSplitBregmanPVC.hpp` | Public header exposing `runSplitBregmanPVC_CUDA()` |
| `src/CudaSplitBregmanPVC.cu` | Device-resident solver loop (~300 lines) |
| `tests/cuda_test.cu` | Smoke test with CPU cross-validation |

## Modified Files (1 file)

| File | Changes |
|------|---------|
| `CMakeLists.txt` | `check_language(CUDA)` guard, appends `.cu` sources when `USE_CUDA=ON`, adds CUDA smoke test target |

## Architectural Decisions

### Constraint 0 — Git branch

Verified on `feature/cuda-backend`. CPU files (`Operators.cpp`, `SplitBregmanPVC.cpp`) are completely untouched.

### Constraint 1 — Zero data ping-pong (Device-Resident State)

All volumes (`u`, `d`, `bregman`, `gv`, scratch buffers) are allocated via `DeviceVolume3D` before the loop. The only `cudaMemcpy` host-to-device is the initial PET/guidance upload; the only device-to-host is the final corrected volume. All iteration work is `<<<kernel>>>` launches and device-to-device copies.

### Constraint 2 — Fused kernels

- **`fusedShrinkBregman`** — combines soft-thresholding shrinkage AND Bregman variable update into one kernel (eliminates 2 extra global memory round-trips)
- **`computeFx2AndPenalty`** — computes `fx2 = d - b_gu - bregman` AND reduces `sum(fx2^2)` using CUB block-level reduction in a single pass
- **`sumDiagonalBackwardDivergence`** — fuses three backward-difference gradients + diagonal summation into one kernel (replaces 3 separate `gradient3DBackward` calls + a summation loop)
- **`lineSearchDots`** — computes `s*y`, `y*y`, `||g||^2` in a single fused pass with CUB block reduction
- **`fusedGradientCombine`** — merges `grad = mu*dfx1 + lambda*dfx2`

### Constraint 3 — Efficient 3D Separable Convolution

Three sequential 1D passes (X, Y, Z) using CUDA constant memory (`__constant__`) for kernel weights. Spatial convolution is used rather than cuFFT because PSF kernels are small (13 taps, `half_width=6`). At this kernel size, the overhead of FFT padding, forward/inverse transforms, and complex-multiply would exceed the cost of direct spatial convolution.

### Constraint 4 — Safe Global Reductions

All global norms and sums use:
- `thrust::transform_reduce` for L2 norms and max-element queries
- `cub::BlockReduce` + `atomicAdd` for fused penalty/dot-product accumulations within compute kernels

No naive atomic reductions over millions of voxels.

### Constraint 5 — Architecture & Fallback

CPU code is completely untouched. Build with `cmake -DUSE_CUDA=ON` to enable the GPU path; default is `OFF`, which builds the original CPU-only code. The `check_language(CUDA)` guard in CMake gracefully degrades with a warning if no CUDA compiler is found.

## GPU Memory Budget

For a volume of size N voxels, the solver allocates approximately **20 volumes x 8 bytes x N** plus small scalar reduction buffers.

| Volume dimensions | N (voxels) | Approximate VRAM |
|-------------------|------------|-------------------|
| 181 x 210 x 181   | ~6.9M      | ~1.1 GB           |
| 256 x 256 x 256   | ~16.8M     | ~2.7 GB           |

## Build & Test

```bash
# CPU only (unchanged)
cmake -B build -DUSE_CUDA=OFF
cmake --build build
ctest --test-dir build

# GPU enabled
cmake -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda
```

## Kernel Launch Configuration

All kernels use a flat 1D grid with 256 threads per block. Grid size is `ceil(N / 256)` where N is the total voxel count. This is sufficient for element-wise operations; the separable convolution kernels run three sequential passes with the same configuration.



# TO REBUILD CUDA BACKEND
"C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" -B build-cuda -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe" -DZLIB_LIBRARY="C:/Users/tihas/miniconda3/Library/lib/zlib.lib" -DZLIB_INCLUDE_DIR="C:/Users/tihas/miniconda3/Library/include"
