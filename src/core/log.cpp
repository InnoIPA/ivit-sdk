/**
 * @file log.cpp
 * @brief Log sink dispatch
 */

#include "ivit/core/log.hpp"
#include "ivit/ivit.hpp"
#include <cstdio>
#include <mutex>

namespace ivit {
namespace {
std::mutex g_sink_mutex;
LogCallback g_sink;

// A sink that logs (or an SDK call made from a sink) must not recurse.
thread_local bool g_in_sink = false;

const char *level_name(LogLevel level) {
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::WARNING:
        return "WARNING";
    case LogLevel::ERROR:
        return "ERROR";
    default:
        return "";
    }
}
} // namespace

void set_log_callback(LogCallback callback) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    g_sink = std::move(callback);
}

namespace detail {
void log(LogLevel level, const std::string &message) {
    const LogLevel threshold = get_parsed_log_level();
    if (level == LogLevel::OFF || threshold == LogLevel::OFF || level < threshold)
        return;
    if (g_in_sink)
        return;

    // Copy so the sink runs outside the lock and may be replaced meanwhile.
    LogCallback sink;
    {
        std::lock_guard<std::mutex> lock(g_sink_mutex);
        sink = g_sink;
    }
    g_in_sink = true;
    try {
        if (sink) {
            sink(level, message);
        } else {
            std::fprintf(stderr, "[iVIT] [%s] %s\n", level_name(level), message.c_str());
        }
    } catch (...) {
        // A faulty sink must never break inference.
    }
    g_in_sink = false;
}
} // namespace detail
} // namespace ivit
