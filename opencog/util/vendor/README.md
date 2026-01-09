# Vendored Dependencies

This directory contains vendored (embedded) copies of essential third-party
dependencies. These provide a self-contained build without requiring system
packages to be installed.

## Structure

```
vendor/
├── cxxtest/         # CXXTest testing framework (trimmed to essentials)
│   ├── *.h          # Header files (17 essential headers)
│   ├── *.cpp        # Implementation files (9 files)
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

### CXXTest (Fully Functional)

The vendored CXXTest is used for running unit tests. It includes:
- Essential CXXTest headers for test development (trimmed from full set)
- The `cxxtestgen` script for generating test runners
- Python modules required by cxxtestgen (Python 2 and 3 support)

**Removed** (not needed for cogutil tests):
- GUI headers (Win32Gui.h, X11Gui.h, QtGui.h, Gui.h)
- Alternative printers (XmlPrinter.h, XUnitPrinter.h, etc.)
- Mock support (Mock.h)

### BFD and libiberty (Reference Headers)

These headers are vendored for reference. The actual BFD and libiberty
**libraries** must still be installed for pretty stack traces.

**Option 1: System packages**
```bash
# Ubuntu/Debian
sudo apt-get install binutils-dev libiberty-dev

# Fedora/RHEL
sudo dnf install binutils-devel
```

**Option 2: Build from 3p sources**
```bash
cd 3p
./build_3p.sh
cd ../build
cmake .. -DUSE_3P_LIBS=ON
```

If the libraries are not available, cogutil will still build and work, but
stack traces will be less detailed.

## Files Included (65 total)

### CXXTest (57 files)
Essential headers and implementations:
- Core: TestSuite.h, TestListener.h, TestRunner.h, TestTracker.h
- Descriptions: Descriptions.h, RealDescriptions.h, DummyDescriptions.h
- Output: ErrorPrinter.h, ErrorFormatter.h
- Support: Flags.h, LinkedList.h, ValueTraits.h, StdHeaders.h
- Root.cpp and supporting .cpp files
- bin/cxxtestgen and python modules

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

## Source Repositories

Full source repositories are available in `3p/` (cloned on demand).
See `3p/README.md` for details on building from source.

## Updating Vendored Dependencies

1. Clone the full repos into `3p/`
2. Copy updated files to `vendor/`
3. Test the build with `make check`
4. Remove any unused files (GUI, alternative printers, etc.)
