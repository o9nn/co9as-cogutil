#
# CMakeLists.txt for third-party dependencies (full source)
#
# This directory contains full source repositories of external dependencies.
# These are used for:
#   1. Building vendored versions of libraries when system packages unavailable
#   2. Reference during development and verification
#   3. Identifying minimal required file sets for the vendor/ directory
#
# Structure:
#   3p/
#   ├── cxxtest/     - CXXTest testing framework (full repo)
#   └── binutils/    - GNU Binutils (includes BFD + libiberty)
#
# Once unified builds are verified successful, we can determine exactly
# which files from these repos are required and trim the vendor/ directory
# to contain only the essential files.
#

option(BUILD_3P_LIBIBERTY "Build libiberty from 3p sources" OFF)
option(BUILD_3P_BFD "Build BFD from 3p sources" OFF)

message(STATUS "Third-party source directory: ${CMAKE_CURRENT_SOURCE_DIR}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cxxtest")
    message(STATUS "  CXXTest: Available")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/binutils")
    message(STATUS "  Binutils: Available (BFD + libiberty)")
endif()

# Building BFD and libiberty from source is complex because they
# require autoconf/configure. For now, we provide the headers in
# vendor/ and link against system libraries.
#
# To build from source, users would need to:
#   cd 3p/binutils
#   mkdir build && cd build
#   ../configure --enable-targets=all --disable-nls
#   make -j$(nproc)
#
# The resulting libraries would be in:
#   3p/binutils/build/bfd/.libs/libbfd.a
#   3p/binutils/build/libiberty/libiberty.a

if(BUILD_3P_LIBIBERTY OR BUILD_3P_BFD)
    message(STATUS "Building 3p libs from source is not yet automated.")
    message(STATUS "Please build binutils manually if system libs unavailable:")
    message(STATUS "  cd ${CMAKE_CURRENT_SOURCE_DIR}/binutils")
    message(STATUS "  mkdir build && cd build")
    message(STATUS "  ../configure --enable-targets=all --disable-nls")
    message(STATUS "  make -j\$(nproc)")
endif()
