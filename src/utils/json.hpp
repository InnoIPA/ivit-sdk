#pragma once

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

namespace ivit::detail {
inline std::string json_string(const std::string &value) {
    std::string result = "\"";
    const char *hex = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}
inline std::string json_number(float value) {
    if (!std::isfinite(value))
        return "null";
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
}
} // namespace ivit::detail
