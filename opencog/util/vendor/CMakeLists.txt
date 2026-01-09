#
# CMakeLists.txt for vendored third-party dependencies
#
# This directory contains vendored copies of external dependencies
# to allow for self-contained builds without system package dependencies.
#
# Structure:
#   vendor/
#   ├── cxxtest/     - CXXTest testing framework (headers + scripts)
#   ├── bfd/         - BFD (Binary File Descriptor) headers from binutils
#   └── libiberty/   - libiberty headers from binutils/GCC
#
# Full source repositories are kept in 3p/ at the project root for
# building the actual libraries when needed.
#

# Export vendor include paths for use by parent CMake
set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

# CXXTest configuration
set(VENDOR_CXXTEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cxxtest" PARENT_SCOPE)
set(VENDOR_CXXTEST_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cxxtest/bin" PARENT_SCOPE)
set(VENDOR_CXXTEST_GEN "${CMAKE_CURRENT_SOURCE_DIR}/cxxtest/bin/cxxtestgen" PARENT_SCOPE)

# BFD/libiberty configuration (headers only - still links against system libs)
set(VENDOR_BFD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bfd" PARENT_SCOPE)
set(VENDOR_IBERTY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libiberty" PARENT_SCOPE)

message(STATUS "Vendor directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  CXXTest headers: ${CMAKE_CURRENT_SOURCE_DIR}/cxxtest")
message(STATUS "  BFD headers: ${CMAKE_CURRENT_SOURCE_DIR}/bfd")
message(STATUS "  libiberty headers: ${CMAKE_CURRENT_SOURCE_DIR}/libiberty")
