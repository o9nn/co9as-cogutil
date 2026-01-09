# Third-Party Dependencies (3p)

This directory contains full source repositories of external dependencies that
cogutil can optionally use. These are cloned on-demand and are **not tracked
in git** (see .gitignore).

## Structure

```
3p/
├── CMakeLists.txt   # Build configuration (tracked)
├── .gitignore       # Excludes cloned repos (tracked)
├── README.md        # This file (tracked)
├── cxxtest/         # CXXTest testing framework (cloned, not tracked)
└── binutils/        # GNU Binutils with BFD + libiberty (cloned, not tracked)
```

## Purpose

1. **Development Reference**: Full source for understanding dependencies
2. **Source Building**: Build dependencies when system packages unavailable
3. **Verification**: Identify minimal required files for `vendor/` directory

## Cloning Dependencies

To clone the dependencies:

```bash
# CXXTest (testing framework)
git clone --depth 1 https://github.com/CxxTest/cxxtest.git 3p/cxxtest

# Binutils (BFD + libiberty for stack traces)
git clone --depth 1 https://sourceware.org/git/binutils-gdb.git 3p/binutils
```

## Building from Source

### Binutils (BFD + libiberty)

If system packages (`binutils-dev`, `libiberty-dev`) are not available:

```bash
cd 3p/binutils
mkdir build && cd build
../configure --enable-targets=all --disable-nls
make -j$(nproc)
```

Libraries will be at:
- `3p/binutils/build/bfd/.libs/libbfd.a`
- `3p/binutils/build/libiberty/libiberty.a`

## Vendored Files

Essential files from these repos are copied to `opencog/util/vendor/` for a
self-contained build without requiring the full cloned repos. The vendor
directory contains only the headers and scripts actually used by cogutil.

See `opencog/util/vendor/` for the minimal dependency set.
