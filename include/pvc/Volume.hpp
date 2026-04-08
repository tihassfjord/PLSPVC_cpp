#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pvc {

class Volume3D {
public:
    Volume3D() = default;

    Volume3D(std::size_t nx, std::size_t ny, std::size_t nz, double value = 0.0)
        : nx_(nx), ny_(ny), nz_(nz), data_(nx * ny * nz, value) {}

    [[nodiscard]] std::size_t nx() const { return nx_; }
    [[nodiscard]] std::size_t ny() const { return ny_; }
    [[nodiscard]] std::size_t nz() const { return nz_; }
    [[nodiscard]] std::size_t size() const { return data_.size(); }

    [[nodiscard]] bool empty() const { return data_.empty(); }

    [[nodiscard]] std::size_t index(std::size_t x, std::size_t y, std::size_t z) const {
        return (z * ny_ + y) * nx_ + x;
    }

    double &operator()(std::size_t x, std::size_t y, std::size_t z) {
        return data_[index(x, y, z)];
    }

    const double &operator()(std::size_t x, std::size_t y, std::size_t z) const {
        return data_[index(x, y, z)];
    }

    [[nodiscard]] const std::vector<double> &data() const { return data_; }
    [[nodiscard]] std::vector<double> &data() { return data_; }

    void fill(double value) { std::fill(data_.begin(), data_.end(), value); }

    [[nodiscard]] double max() const {
        if (data_.empty()) {
            return 0.0;
        }
        return *std::max_element(data_.begin(), data_.end());
    }

    [[nodiscard]] double min() const {
        if (data_.empty()) {
            return 0.0;
        }
        return *std::min_element(data_.begin(), data_.end());
    }

    [[nodiscard]] double l2Norm() const {
        long double sum = 0.0L;
        for (double v : data_) {
            sum += static_cast<long double>(v) * static_cast<long double>(v);
        }
        return std::sqrt(static_cast<double>(sum));
    }

    [[nodiscard]] double sum() const {
        long double acc = 0.0L;
        for (double v : data_) {
            acc += v;
        }
        return static_cast<double>(acc);
    }

    void scaleInPlace(double scale) {
        for (double &v : data_) {
            v *= scale;
        }
    }

    void addScaledInPlace(const Volume3D &other, double alpha) {
        requireSameShape(other, "addScaledInPlace");
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] += alpha * other.data_[i];
        }
    }

    [[nodiscard]] bool sameShape(const Volume3D &other) const {
        return nx_ == other.nx_ && ny_ == other.ny_ && nz_ == other.nz_;
    }

    void requireSameShape(const Volume3D &other, const std::string &context) const {
        if (!sameShape(other)) {
            throw std::runtime_error("Shape mismatch in " + context);
        }
    }

private:
    std::size_t nx_ = 0;
    std::size_t ny_ = 0;
    std::size_t nz_ = 0;
    std::vector<double> data_;
};

struct VectorField3D {
    Volume3D x;
    Volume3D y;
    Volume3D z;

    VectorField3D() = default;

    explicit VectorField3D(const Volume3D &reference)
        : x(reference.nx(), reference.ny(), reference.nz(), 0.0),
          y(reference.nx(), reference.ny(), reference.nz(), 0.0),
          z(reference.nx(), reference.ny(), reference.nz(), 0.0) {}

    VectorField3D(std::size_t nx, std::size_t ny, std::size_t nz)
        : x(nx, ny, nz, 0.0), y(nx, ny, nz, 0.0), z(nx, ny, nz, 0.0) {}

    [[nodiscard]] std::size_t size() const { return x.size(); }
};

struct NiftiImage {
    std::vector<int> dims;     // dims[0] = ndims, dims[1..7] sizes
    std::vector<double> pixdim; // pixdim[1..7]
    std::vector<double> data;  // Always converted to double on read
    int datatype = 0;
    bool little_endian = true;

    [[nodiscard]] bool is3D() const {
        return dims.size() >= 4 && dims[0] >= 3 && dims[1] > 0 && dims[2] > 0 && dims[3] > 0;
    }

    [[nodiscard]] bool is4D() const {
        return dims.size() >= 5 && dims[0] >= 4 && dims[4] > 0;
    }

    [[nodiscard]] std::size_t nx() const { return static_cast<std::size_t>(dims.at(1)); }
    [[nodiscard]] std::size_t ny() const { return static_cast<std::size_t>(dims.at(2)); }
    [[nodiscard]] std::size_t nz() const { return static_cast<std::size_t>(dims.at(3)); }
    [[nodiscard]] std::size_t nt() const {
        if (dims.size() >= 5 && dims[0] >= 4) {
            return static_cast<std::size_t>(std::max(dims.at(4), 1));
        }
        return 1;
    }

    [[nodiscard]] std::size_t voxelCount3D() const {
        return nx() * ny() * nz();
    }

    [[nodiscard]] std::size_t voxelCountTotal() const {
        return voxelCount3D() * nt();
    }

    [[nodiscard]] std::size_t index(std::size_t x, std::size_t y, std::size_t z, std::size_t t = 0) const {
        return (((t * nz() + z) * ny()) + y) * nx() + x;
    }

    [[nodiscard]] Volume3D frame(std::size_t t = 0) const {
        if (!is3D()) {
            throw std::runtime_error("NIfTI image is not at least 3D.");
        }
        if (t >= nt()) {
            throw std::runtime_error("Requested time frame is out of range.");
        }
        Volume3D out(nx(), ny(), nz(), 0.0);
        const std::size_t frame_voxels = voxelCount3D();
        const std::size_t offset = t * frame_voxels;
        std::copy(data.begin() + static_cast<std::ptrdiff_t>(offset),
                  data.begin() + static_cast<std::ptrdiff_t>(offset + frame_voxels),
                  out.data().begin());
        return out;
    }

    [[nodiscard]] Volume3D meanOverTime() const {
        if (!is3D()) {
            throw std::runtime_error("NIfTI image is not at least 3D.");
        }
        Volume3D out(nx(), ny(), nz(), 0.0);
        const std::size_t frame_voxels = voxelCount3D();
        const std::size_t frames = nt();
        if (frames == 1) {
            std::copy(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(frame_voxels), out.data().begin());
            return out;
        }
        for (std::size_t t = 0; t < frames; ++t) {
            const std::size_t offset = t * frame_voxels;
            for (std::size_t i = 0; i < frame_voxels; ++i) {
                out.data()[i] += data[offset + i];
            }
        }
        const double inv_frames = 1.0 / static_cast<double>(frames);
        for (double &v : out.data()) {
            v *= inv_frames;
        }
        return out;
    }
};

inline double safeDivide(double numerator, double denominator, double fallback = 0.0) {
    if (std::abs(denominator) <= std::numeric_limits<double>::epsilon()) {
        return fallback;
    }
    return numerator / denominator;
}

} // namespace pvc
