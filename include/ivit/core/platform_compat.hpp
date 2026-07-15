#pragma once
// Cross-platform shims for POSIX file-descriptor and environment APIs.
// On Windows (MSVC) these POSIX names do not exist; map them to the
// underscore-prefixed CRT equivalents. On POSIX use the standard names.

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>

#ifdef _WIN32
#include <io.h>
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#define IVIT_DEVNULL "NUL"
#define IVIT_STDERR_FILENO _fileno(stderr)
inline int ivit_dup(int fd) { return _dup(fd); }
inline int ivit_dup2(int oldfd, int newfd) { return _dup2(oldfd, newfd); }
inline int ivit_open(const char* path, int flags) { return _open(path, flags); }
inline int ivit_close(int fd) { return _close(fd); }
inline int ivit_setenv(const char* name, const char* value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
#else
#include <unistd.h>
#define IVIT_DEVNULL "/dev/null"
#define IVIT_STDERR_FILENO STDERR_FILENO
inline int ivit_dup(int fd) { return dup(fd); }
inline int ivit_dup2(int oldfd, int newfd) { return dup2(oldfd, newfd); }
inline int ivit_open(const char* path, int flags) { return open(path, flags); }
inline int ivit_close(int fd) { return close(fd); }
inline int ivit_setenv(const char* name, const char* value, int overwrite) {
    return setenv(name, value, overwrite);
}
#endif
