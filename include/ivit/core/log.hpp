/**
 * @file log.hpp
 * @brief Logging levels and an application-supplied log sink.
 *
 * By default every SDK message is written to stderr (never stdout, so the
 * SDK cannot corrupt an application's data stream). Call set_log_callback()
 * to route messages into the host application's own logging.
 */

#ifndef IVIT_CORE_LOG_HPP
#define IVIT_CORE_LOG_HPP

#include <functional>
#include <string>

namespace ivit {

/**
 * @brief Log level enumeration (ordered by increasing severity; OFF disables).
 */
enum class LogLevel {
    DEBUG,   ///< Verbose debug messages
    INFO,    ///< Informational messages
    WARNING, ///< Warning messages
    ERROR,   ///< Error messages
    OFF      ///< No logging
};

/**
 * @brief Log sink signature.
 *
 * Called for every message that passes the current log level. May be called
 * from any SDK or backend thread, possibly concurrently; the callback must be
 * thread-safe, must not throw (exceptions are swallowed) and must not call
 * back into the SDK. Messages carry no trailing newline.
 */
using LogCallback = std::function<void(LogLevel level, const std::string &message)>;

/**
 * @brief Install a log sink, or pass an empty function to restore stderr.
 *
 * Install the callback before loading models. Clear it before the objects it
 * references are destroyed; the SDK may log while the process exits.
 */
void set_log_callback(LogCallback callback);

namespace detail {
/// Internal: filter by the current level and deliver to the sink.
void log(LogLevel level, const std::string &message);
} // namespace detail

} // namespace ivit

#endif // IVIT_CORE_LOG_HPP
