# Vendored Dependencies

This directory contains vendored (embedded) copies of essential third-party
dependencies. These provide a self-contained build without requiring system
packages to be installed.

## Structure

```
vendor/
├── cxxtest/         # CXXTest testing framework
│   ├── *.h          # Header files
│   ├── *.cpp        # Implementation files
│   ├── bin/         # cxxtestgen script
│   └── python/      # Python modules for test generation
├── bfd/             # BFD (Binary File Descriptor) headers
│   └── bfd.h, etc.  # Headers from GNU binutils
└── libiberty/       # libiberty headers
    └── libiberty.h  # Header from GNU binutils/GCC
```

## Usage

The vendored dependencies are automatically detected and used by CMake when
building cogutil. No additional configuration is required.

### CXXTest

The vendored CXXTest is used for running unit tests. It includes:
- All CXXTest headers for test development
- The `cxxtestgen` script for generating test runners
- Python modules required by cxxtestgen

### BFD and libiberty

These headers are vendored for reference and for systems where binutils-dev
and libiberty-dev packages are not available. The actual BFD and libiberty
**libraries** must still be installed on the system for pretty stack traces:

```bash
# Ubuntu/Debian
sudo apt-get install binutils-dev libiberty-dev

# Fedora/RHEL
sudo dnf install binutils-devel
```

If the libraries are not available, cogutil will still build and work, but
stack traces will be less detailed.

## Source Repositories

The full source repositories for these dependencies are available in the
`3p/` directory (cloned on demand, not tracked in git). See `3p/README.md`
for details on building from source.

## Files Included

### CXXTest (72 files)
- Headers: TestSuite.h, TestListener.h, TestRunner.h, etc.
- Implementation: Root.cpp, GlobalFixture.cpp, etc.
- Scripts: bin/cxxtestgen
- Python: python/cxxtest/*.py

### BFD (5 files)
- bfd.h (from bfd-in2.h)
- bfdlink.h
- ansidecl.h
- diagnostics.h
- symcat.h

### libiberty (3 files)
- libiberty.h
- demangle.h
- ansidecl.h

## Updating Vendored Dependencies

To update the vendored files from newer versions:

1. Clone the full repos into `3p/`
2. Copy updated files to `vendor/`
3. Test the build with `make check`
