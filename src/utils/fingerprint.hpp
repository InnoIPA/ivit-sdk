#pragma once

#include "ivit/core/common.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

namespace ivit::detail {

// Incremental SHA-256 keeps model hashing bounded in memory, without adding a
// mandatory crypto library to downstream SDK consumers.
class Sha256 {
  public:
    void update(const void *bytes, size_t count) {
        auto data = static_cast<const uint8_t *>(bytes);
        length_ += count;
        while (count--) {
            block_[used_++] = *data++;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    std::string finish() {
        const uint64_t bits = length_ * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            while (used_ < 64)
                block_[used_++] = 0;
            compress();
            used_ = 0;
        }
        while (used_ < 56)
            block_[used_++] = 0;
        for (int i = 7; i >= 0; --i)
            block_[used_++] = uint8_t(bits >> (i * 8));
        compress();
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (auto value : state_)
            out << std::setw(8) << value;
        return out.str();
    }

  private:
    static uint32_t rotate(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(block_[4 * i]) << 24) | (uint32_t(block_[4 * i + 1]) << 16) |
                   (uint32_t(block_[4 * i + 2]) << 8) | block_[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const auto a = w[i - 15], b = w[i - 2];
            w[i] = w[i - 16] + (rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3)) + w[i - 7] +
                   (rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10));
        }
        auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const auto t1 = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) +
                            ((e & f) ^ ((~e) & g)) + k[i] + w[i];
            const auto t2 =
                (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
    std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t length_ = 0;
};

inline std::string fingerprint(const std::string &value) {
    Sha256 hash;
    hash.update(value.data(), value.size());
    return hash.finish();
}

inline std::string file_fingerprint(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw ModelLoadError("Cannot read model: " + path.string());
    Sha256 hash;
    std::array<char, 65536> buffer;
    while (file) {
        file.read(buffer.data(), buffer.size());
        hash.update(buffer.data(), static_cast<size_t>(file.gcount()));
    }
    if (!file.eof())
        throw ModelLoadError("Cannot hash model: " + path.string());
    return hash.finish();
}

inline std::string model_fingerprint(const std::string &path) {
    namespace fs = std::filesystem;
    auto source = fs::weakly_canonical(fs::absolute(path));
    std::string key = source.string() + "\n" + file_fingerprint(source);
    if (source.extension() == ".xml") {
        source.replace_extension(".bin");
        key += "\n" + file_fingerprint(source);
    }
    return fingerprint(key);
}

inline std::string contract_fingerprint(const LoadConfig &c) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(9);
    auto field = [&out](const std::string &value) { out << value.size() << ':' << value; };
    for (const auto &value : {c.device, c.backend, c.task, c.precision, c.decoder, c.input_color,
                              c.input_layout, c.resize_mode, c.output_name, c.cache_dir})
        field(value);
    out << c.batch_size << ':' << c.input_scale << ':';
    out << c.max_batch_size << ':' << c.num_threads << ':' << c.enable_profiling << ':'
        << c.workspace_size << ':' << c.num_instances << ':';
    field(c.performance_mode);
    for (const auto &values : {c.mean, c.std}) {
        out << values.size() << ':';
        for (float v : values)
            out << v << ':';
    }
    out << c.labels.size() << ':';
    for (const auto &label : c.labels)
        field(label);
    return fingerprint(out.str());
}

} // namespace ivit::detail
