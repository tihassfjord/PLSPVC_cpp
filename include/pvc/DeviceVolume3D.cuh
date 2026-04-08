#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace pvc {
namespace cuda {

// ---------------------------------------------------------------------------
// Thin RAII wrapper around a device-allocated 3-D volume of doubles.
// Movable, non-copyable.  cudaFree is called in the destructor.
// ---------------------------------------------------------------------------
struct DeviceVolume3D {
    double *ptr   = nullptr;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;

    DeviceVolume3D() = default;

    DeviceVolume3D(std::size_t nx_, std::size_t ny_, std::size_t nz_)
        : nx(nx_), ny(ny_), nz(nz_)
    {
        const std::size_t bytes = nx * ny * nz * sizeof(double);
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }

    ~DeviceVolume3D() { reset(); }

    // Move
    DeviceVolume3D(DeviceVolume3D &&o) noexcept
        : ptr(o.ptr), nx(o.nx), ny(o.ny), nz(o.nz)
    {
        o.ptr = nullptr;
        o.nx = o.ny = o.nz = 0;
    }
    DeviceVolume3D &operator=(DeviceVolume3D &&o) noexcept {
        if (this != &o) {
            reset();
            ptr = o.ptr; nx = o.nx; ny = o.ny; nz = o.nz;
            o.ptr = nullptr; o.nx = o.ny = o.nz = 0;
        }
        return *this;
    }

    // No copy
    DeviceVolume3D(const DeviceVolume3D &) = delete;
    DeviceVolume3D &operator=(const DeviceVolume3D &) = delete;

    [[nodiscard]] std::size_t size()  const { return nx * ny * nz; }
    [[nodiscard]] std::size_t bytes() const { return size() * sizeof(double); }

    void reset() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
        nx = ny = nz = 0;
    }

    // Zero all elements on device
    void zero() { cudaMemset(ptr, 0, bytes()); }

    // Host ↔ Device transfers
    void copyFromHost(const double *host_ptr) {
        cudaMemcpy(ptr, host_ptr, bytes(), cudaMemcpyHostToDevice);
    }
    void copyFromHost(const std::vector<double> &v) {
        copyFromHost(v.data());
    }
    void copyToHost(double *host_ptr) const {
        cudaMemcpy(host_ptr, ptr, bytes(), cudaMemcpyDeviceToHost);
    }
    void copyToHost(std::vector<double> &v) const {
        v.resize(size());
        copyToHost(v.data());
    }

    // Device ↔ Device
    void copyFrom(const DeviceVolume3D &src) {
        cudaMemcpy(ptr, src.ptr, bytes(), cudaMemcpyDeviceToDevice);
    }
};

// Three-component vector field on device (analogous to VectorField3D).
struct DeviceVectorField3D {
    DeviceVolume3D x, y, z;

    DeviceVectorField3D() = default;

    DeviceVectorField3D(std::size_t nx, std::size_t ny, std::size_t nz)
        : x(nx, ny, nz), y(nx, ny, nz), z(nx, ny, nz) {}

    [[nodiscard]] std::size_t size() const { return x.size(); }

    void zero() { x.zero(); y.zero(); z.zero(); }

    void copyFrom(const DeviceVectorField3D &src) {
        x.copyFrom(src.x);
        y.copyFrom(src.y);
        z.copyFrom(src.z);
    }
};

// ---------------------------------------------------------------------------
// Helper: check last CUDA error (used after kernel launches in debug builds)
// ---------------------------------------------------------------------------
inline void checkCuda(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace pvc
