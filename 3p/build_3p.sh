#!/bin/bash
#
# Build script for third-party dependencies from source
#
# This script builds libiberty and BFD from the vendored binutils sources.
# Use this when system packages (binutils-dev, libiberty-dev) are not available.
#
# Usage:
#   ./build_3p.sh [all|libiberty|bfd|clean]
#
# After building, reconfigure cogutil with:
#   cmake .. -DUSE_3P_LIBS=ON
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINUTILS_DIR="$SCRIPT_DIR/binutils"
BUILD_DIR="$BINUTILS_DIR/build"

if [ ! -d "$BINUTILS_DIR" ]; then
    echo "Error: binutils source not found at $BINUTILS_DIR"
    echo "Please clone it first:"
    echo "  git clone --depth 1 https://sourceware.org/git/binutils-gdb.git $BINUTILS_DIR"
    exit 1
fi

build_libs() {
    echo "=== Building binutils (libiberty + BFD) ==="
    echo "Source: $BINUTILS_DIR"
    echo "Build:  $BUILD_DIR"

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if [ ! -f Makefile ]; then
        echo "=== Configuring binutils ==="
        # Build with -fPIC so libraries can be linked into shared objects
        CFLAGS="-fPIC" CXXFLAGS="-fPIC" ../configure \
            --enable-targets=all \
            --disable-nls \
            --disable-gdb \
            --disable-gdbserver \
            --disable-sim \
            --disable-readline \
            --disable-libdecnumber \
            --disable-libbacktrace \
            --disable-gold \
            --disable-ld \
            --disable-gprofng \
            --disable-gas \
            --disable-binutils \
            --quiet
    fi

    # Configure subdirectories (needed before building)
    echo "=== Configuring subdirectories ==="
    make configure-libiberty configure-bfd configure-zlib configure-libsframe

    echo "=== Building libiberty ==="
    make -C libiberty -j$(nproc)

    echo "=== Building zlib ==="
    make -C zlib libz.a -j$(nproc)

    echo "=== Building libsframe ==="
    make -C libsframe -j$(nproc)

    echo "=== Building BFD ==="
    make -C bfd libbfd.la -j$(nproc)

    echo ""
    echo "=== Build complete ==="
    echo "Libraries built:"
    echo "  libiberty: $BUILD_DIR/libiberty/libiberty.a"
    echo "  libsframe: $BUILD_DIR/libsframe/.libs/libsframe.a"
    echo "  zlib:      $BUILD_DIR/zlib/libz.a"
    echo "  BFD:       $BUILD_DIR/bfd/.libs/libbfd.a"
    echo ""
    echo "Headers available:"
    echo "  libiberty: $BINUTILS_DIR/include/libiberty.h"
    echo "  BFD:       $BUILD_DIR/bfd/bfd.h"
    echo ""
    echo "To use these libraries, reconfigure cogutil with:"
    echo "  cmake .. -DUSE_3P_LIBS=ON"
}

clean_build() {
    echo "=== Cleaning build directory ==="
    rm -rf "$BUILD_DIR"
    echo "Cleaned: $BUILD_DIR"
}

case "${1:-all}" in
    all|libiberty|bfd)
        build_libs
        ;;
    clean)
        clean_build
        ;;
    *)
        echo "Usage: $0 [all|libiberty|bfd|clean]"
        echo ""
        echo "Commands:"
        echo "  all       Build all (libiberty + BFD)"
        echo "  libiberty Build libiberty only"
        echo "  bfd       Build BFD only"
        echo "  clean     Remove build directory"
        exit 1
        ;;
esac
