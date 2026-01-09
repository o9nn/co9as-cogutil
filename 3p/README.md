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
├── build_3p.sh      # Automated build script (tracked)
├── cxxtest/         # CXXTest testing framework (cloned, not tracked)
└── binutils/        # GNU Binutils with BFD + libiberty (cloned, not tracked)
```

## Purpose

1. **Development Reference**: Full source for understanding dependencies
2. **Source Building**: Build dependencies when system packages unavailable
3. **Verification**: Identify minimal required files for `vendor/` directory

## Quick Start

### Clone Dependencies

```bash
# CXXTest (testing framework) - Optional, already vendored
git clone --depth 1 https://github.com/CxxTest/cxxtest.git 3p/cxxtest

# Binutils (BFD + libiberty for stack traces) - Required for build
git clone --depth 1 https://sourceware.org/git/binutils-gdb.git 3p/binutils
```

### Build from Source

Use the automated build script:

```bash
cd 3p
./build_3p.sh        # Build BFD + libiberty
./build_3p.sh clean  # Clean build directory
```

### Configure cogutil to Use 3p Libraries

After building, reconfigure cogutil:

```bash
cd build
cmake .. -DUSE_3P_LIBS=ON
make
```

## Manual Building

If you prefer manual building:

```bash
cd 3p/binutils
mkdir build && cd build
../configure --enable-targets=all --disable-nls --disable-gdb \
             --disable-gdbserver --disable-sim --disable-gold \
             --disable-ld --disable-gprofng --disable-gas --disable-binutils
make -C libiberty -j$(nproc)
make -C bfd -j$(nproc)
```

Libraries will be at:
- `3p/binutils/build/libiberty/libiberty.a`
- `3p/binutils/build/bfd/.libs/libbfd.a`

## Vendored Files

Essential files from these repos are copied to `opencog/util/vendor/` for a
self-contained build without requiring the full cloned repos. The vendor
directory contains only the headers and scripts actually used by cogutil.

### CXXTest (Fully Vendored)
- Headers, scripts, and Python modules are in `opencog/util/vendor/cxxtest/`
- No system installation required - works out of the box

### BFD/libiberty (Headers Only)
- Reference headers are in `opencog/util/vendor/bfd/` and `libiberty/`
- Actual libraries still required (system packages or 3p build)
- System packages: `apt install binutils-dev libiberty-dev`
- Or use: `./build_3p.sh` + `cmake .. -DUSE_3P_LIBS=ON`

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_3P_LIBS` | OFF | Use libraries built from 3p sources |
