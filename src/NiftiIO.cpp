#include "pvc/NiftiIO.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <zlib.h>

namespace pvc {
namespace {

#pragma pack(push, 1)
struct Nifti1Header {
    int32_t sizeof_hdr;
    char data_type[10];
    char db_name[18];
    int32_t extents;
    int16_t session_error;
    char regular;
    char dim_info;
    int16_t dim[8];
    float intent_p1;
    float intent_p2;
    float intent_p3;
    int16_t intent_code;
    int16_t datatype;
    int16_t bitpix;
    int16_t slice_start;
    float pixdim[8];
    float vox_offset;
    float scl_slope;
    float scl_inter;
    int16_t slice_end;
    char slice_code;
    char xyzt_units;
    float cal_max;
    float cal_min;
    float slice_duration;
    float toffset;
    int32_t glmax;
    int32_t glmin;
    char descrip[80];
    char aux_file[24];
    int16_t qform_code;
    int16_t sform_code;
    float quatern_b;
    float quatern_c;
    float quatern_d;
    float qoffset_x;
    float qoffset_y;
    float qoffset_z;
    float srow_x[4];
    float srow_y[4];
    float srow_z[4];
    char intent_name[16];
    char magic[4];
};
#pragma pack(pop)

static_assert(sizeof(Nifti1Header) == 348, "Unexpected NIfTI-1 header size.");

template <typename T>
T byteSwap(T value) {
    static_assert(std::is_trivially_copyable<T>::value, "byteSwap requires trivially copyable types.");
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    std::reverse(bytes.begin(), bytes.end());
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

void swapHeaderEndianness(Nifti1Header &hdr) {
    hdr.sizeof_hdr = byteSwap(hdr.sizeof_hdr);
    hdr.extents = byteSwap(hdr.extents);
    hdr.session_error = byteSwap(hdr.session_error);
    for (int i = 0; i < 8; ++i) {
        hdr.dim[i] = byteSwap(hdr.dim[i]);
        hdr.pixdim[i] = byteSwap(hdr.pixdim[i]);
    }
    hdr.intent_p1 = byteSwap(hdr.intent_p1);
    hdr.intent_p2 = byteSwap(hdr.intent_p2);
    hdr.intent_p3 = byteSwap(hdr.intent_p3);
    hdr.intent_code = byteSwap(hdr.intent_code);
    hdr.datatype = byteSwap(hdr.datatype);
    hdr.bitpix = byteSwap(hdr.bitpix);
    hdr.slice_start = byteSwap(hdr.slice_start);
    hdr.vox_offset = byteSwap(hdr.vox_offset);
    hdr.scl_slope = byteSwap(hdr.scl_slope);
    hdr.scl_inter = byteSwap(hdr.scl_inter);
    hdr.slice_end = byteSwap(hdr.slice_end);
    hdr.cal_max = byteSwap(hdr.cal_max);
    hdr.cal_min = byteSwap(hdr.cal_min);
    hdr.slice_duration = byteSwap(hdr.slice_duration);
    hdr.toffset = byteSwap(hdr.toffset);
    hdr.glmax = byteSwap(hdr.glmax);
    hdr.glmin = byteSwap(hdr.glmin);
    hdr.qform_code = byteSwap(hdr.qform_code);
    hdr.sform_code = byteSwap(hdr.sform_code);
    hdr.quatern_b = byteSwap(hdr.quatern_b);
    hdr.quatern_c = byteSwap(hdr.quatern_c);
    hdr.quatern_d = byteSwap(hdr.quatern_d);
    hdr.qoffset_x = byteSwap(hdr.qoffset_x);
    hdr.qoffset_y = byteSwap(hdr.qoffset_y);
    hdr.qoffset_z = byteSwap(hdr.qoffset_z);
    for (int i = 0; i < 4; ++i) {
        hdr.srow_x[i] = byteSwap(hdr.srow_x[i]);
        hdr.srow_y[i] = byteSwap(hdr.srow_y[i]);
        hdr.srow_z[i] = byteSwap(hdr.srow_z[i]);
    }
}

bool endsWith(const std::string &text, const std::string &suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::uint8_t> readAllBytes(const std::string &path) {
    if (endsWith(path, ".gz")) {
        gzFile file = gzopen(path.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("Failed to open gzip NIfTI file: " + path);
        }
        std::vector<std::uint8_t> buffer;
        std::array<char, 1 << 15> chunk{};
        int bytes_read = 0;
        while ((bytes_read = gzread(file, chunk.data(), static_cast<unsigned int>(chunk.size()))) > 0) {
            buffer.insert(buffer.end(), reinterpret_cast<std::uint8_t *>(chunk.data()),
                          reinterpret_cast<std::uint8_t *>(chunk.data()) + bytes_read);
        }
        gzclose(file);
        return buffer;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open NIfTI file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("NIfTI file appears to be empty: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

void writeAllBytes(const std::string &path, const std::vector<std::uint8_t> &bytes) {
    if (endsWith(path, ".gz")) {
        gzFile file = gzopen(path.c_str(), "wb");
        if (!file) {
            throw std::runtime_error("Failed to open gzip output path: " + path);
        }
        const int written = gzwrite(file, bytes.data(), static_cast<unsigned int>(bytes.size()));
        gzclose(file);
        if (written == 0) {
            throw std::runtime_error("Failed to write gzip NIfTI file: " + path);
        }
        return;
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open output path: " + path);
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::size_t bytesPerVoxel(int datatype) {
    switch (datatype) {
    case 2:   return 1; // uint8
    case 4:   return 2; // int16
    case 8:   return 4; // int32
    case 16:  return 4; // float32
    case 64:  return 8; // float64
    case 256: return 1; // int8
    case 512: return 2; // uint16
    case 768: return 4; // uint32
    case 1024:return 8; // int64
    case 1280:return 8; // uint64
    default:
        throw std::runtime_error("Unsupported NIfTI datatype code: " + std::to_string(datatype));
    }
}

template <typename T>
void readTypedToDouble(const std::uint8_t *src, std::size_t count, bool swap_endian,
                       double scl_slope, double scl_inter, std::vector<double> &dst) {
    dst.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        T value{};
        std::memcpy(&value, src + i * sizeof(T), sizeof(T));
        if (swap_endian && sizeof(T) > 1) {
            value = byteSwap(value);
        }
        double converted = static_cast<double>(value);
        if (std::abs(scl_slope) > std::numeric_limits<double>::epsilon()) {
            converted = converted * scl_slope + scl_inter;
        }
        dst[i] = converted;
    }
}

void copyHeaderToImage(const Nifti1Header &hdr, bool little_endian, NiftiImage &image) {
    image.dims.assign(8, 1);
    image.pixdim.assign(8, 1.0);
    for (int i = 0; i < 8; ++i) {
        image.dims[i] = hdr.dim[i];
        image.pixdim[i] = hdr.pixdim[i];
    }
    image.datatype = hdr.datatype;
    image.little_endian = little_endian;
    // Store original header so we can reuse its spatial transform matrices later
    std::memcpy(image.original_header.data(), &hdr, sizeof(Nifti1Header));
}

std::vector<std::uint8_t> packDoubleData(const std::vector<double> &data) {
    std::vector<std::uint8_t> out(data.size() * sizeof(double));
    std::memcpy(out.data(), data.data(), out.size());
    return out;
}

Nifti1Header buildHeaderForOutput(const NiftiImage &reference_header,
                                  std::size_t nx,
                                  std::size_t ny,
                                  std::size_t nz,
                                  std::size_t nt,
                                  const char *description) {
    Nifti1Header hdr{};
    
    // Copy the exact affine transforms from the original header
    std::memcpy(&hdr, reference_header.original_header.data(), sizeof(Nifti1Header));

    hdr.dim[0] = static_cast<int16_t>(nt > 1 ? 4 : 3);
    hdr.dim[1] = static_cast<int16_t>(nx);
    hdr.dim[2] = static_cast<int16_t>(ny);
    hdr.dim[3] = static_cast<int16_t>(nz);
    hdr.dim[4] = static_cast<int16_t>(nt);
    hdr.dim[5] = 1;
    hdr.dim[6] = 1;
    hdr.dim[7] = 1;
    hdr.datatype = 64;  // Float64 / double
    hdr.bitpix = 64;
    
    // Ensure scaling parameters are flat since data is saved directly as double
    hdr.scl_slope = 1.0f;
    hdr.scl_inter = 0.0f;
    
    std::snprintf(hdr.descrip, sizeof(hdr.descrip), "%s", description);
    return hdr;
}

void validateShapeAgainstReference(const NiftiImage &reference_header,
                                   const Volume3D &volume,
                                   const std::string &context) {
    if (!reference_header.is3D()) {
        throw std::runtime_error("Reference NIfTI header must be at least 3D.");
    }
    if (volume.nx() != reference_header.nx() ||
        volume.ny() != reference_header.ny() ||
        volume.nz() != reference_header.nz()) {
        throw std::runtime_error("Output volume shape does not match the reference NIfTI header in " + context + ".");
    }
}

} // namespace

NiftiImage readNifti(const std::string &path) {
    const std::vector<std::uint8_t> bytes = readAllBytes(path);
    if (bytes.size() < sizeof(Nifti1Header)) {
        throw std::runtime_error("File is too small to be a valid NIfTI-1 file: " + path);
    }

    Nifti1Header hdr{};
    std::memcpy(&hdr, bytes.data(), sizeof(Nifti1Header));

    bool little_endian = true;
    if (hdr.sizeof_hdr != 348) {
        swapHeaderEndianness(hdr);
        if (hdr.sizeof_hdr != 348) {
            throw std::runtime_error("Could not parse NIfTI header endianness for: " + path);
        }
        little_endian = false;
    }

    if (!(std::strncmp(hdr.magic, "n+1", 3) == 0 || std::strncmp(hdr.magic, "ni1", 3) == 0)) {
        throw std::runtime_error("Only NIfTI-1 files are supported (unexpected magic field).");
    }
    if (std::strncmp(hdr.magic, "ni1", 3) == 0) {
        throw std::runtime_error("Two-file .hdr/.img NIfTI is not supported by this minimal reader.");
    }

    NiftiImage image;
    copyHeaderToImage(hdr, little_endian, image);
    if (!image.is3D()) {
        throw std::runtime_error("NIfTI image must be at least 3D.");
    }

    const std::size_t total_voxels = image.voxelCountTotal();
    const std::size_t bpp = bytesPerVoxel(hdr.datatype);
    const std::size_t offset = static_cast<std::size_t>(std::max(hdr.vox_offset, 352.0f));
    const std::size_t required = offset + total_voxels * bpp;
    if (required > bytes.size()) {
        throw std::runtime_error("NIfTI file ended before the expected voxel data was fully available.");
    }

    const double scl_slope = static_cast<double>(hdr.scl_slope);
    const double scl_inter = static_cast<double>(hdr.scl_inter);
    const std::uint8_t *src = bytes.data() + offset;
    const bool swap_endian = !little_endian;

    switch (hdr.datatype) {
    case 2:
        readTypedToDouble<std::uint8_t>(src, total_voxels, false, scl_slope, scl_inter, image.data);
        break;
    case 4:
        readTypedToDouble<std::int16_t>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 8:
        readTypedToDouble<std::int32_t>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 16:
        readTypedToDouble<float>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 64:
        readTypedToDouble<double>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 256:
        readTypedToDouble<std::int8_t>(src, total_voxels, false, scl_slope, scl_inter, image.data);
        break;
    case 512:
        readTypedToDouble<std::uint16_t>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 768:
        readTypedToDouble<std::uint32_t>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 1024:
        readTypedToDouble<std::int64_t>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    case 1280:
        readTypedToDouble<std::uint64_t>(src, total_voxels, swap_endian, scl_slope, scl_inter, image.data);
        break;
    default:
        throw std::runtime_error("Unsupported datatype encountered while reading NIfTI.");
    }

    return image;
}

void writeNifti(const std::string &path, const NiftiImage &reference_header, const Volume3D &volume) {
    validateShapeAgainstReference(reference_header, volume, "writeNifti(3D)");

    Nifti1Header hdr = buildHeaderForOutput(reference_header,
                                            volume.nx(), volume.ny(), volume.nz(), 1,
                                            "PLS PVC output (double)");

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(hdr.vox_offset), 0);
    std::memcpy(bytes.data(), &hdr, sizeof(Nifti1Header));

    const std::vector<std::uint8_t> data_bytes = packDoubleData(volume.data());
    bytes.insert(bytes.end(), data_bytes.begin(), data_bytes.end());
    writeAllBytes(path, bytes);
}

void writeNifti(const std::string &path, const NiftiImage &reference_header, const std::vector<Volume3D> &volumes) {
    if (volumes.empty()) {
        throw std::runtime_error("Cannot write an empty 4D NIfTI series.");
    }
    for (const auto &frame : volumes) {
        validateShapeAgainstReference(reference_header, frame, "writeNifti(4D)");
    }

    Nifti1Header hdr = buildHeaderForOutput(reference_header,
                                            volumes.front().nx(), volumes.front().ny(), volumes.front().nz(),
                                            volumes.size(),
                                            "PLS PVC 4D output (double)");

    std::vector<double> packed;
    packed.reserve(volumes.size() * volumes.front().size());
    for (const auto &frame : volumes) {
        packed.insert(packed.end(), frame.data().begin(), frame.data().end());
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(hdr.vox_offset), 0);
    std::memcpy(bytes.data(), &hdr, sizeof(Nifti1Header));

    const std::vector<std::uint8_t> data_bytes = packDoubleData(packed);
    bytes.insert(bytes.end(), data_bytes.begin(), data_bytes.end());
    writeAllBytes(path, bytes);
}

} // namespace pvc
