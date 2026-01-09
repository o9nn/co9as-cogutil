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
#   ├── binutils/    - GNU Binutils (includes BFD + libiberty)
#   └── build_3p.sh  - Build script for dependencies
#

message(STATUS "Third-party source directory: ${CMAKE_CURRENT_SOURCE_DIR}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cxxtest")
    message(STATUS "  CXXTest: Available (vendored in opencog/util/vendor)")
endif()

set(_3P_LIBS_AVAILABLE FALSE)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/binutils")
    message(STATUS "  Binutils: Available (BFD + libiberty source)")

    # Check for pre-built libraries
    set(3P_BINUTILS_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/binutils/build")
    set(3P_LIBIBERTY_A "${3P_BINUTILS_BUILD}/libiberty/libiberty.a")
    set(3P_BFD_A "${3P_BINUTILS_BUILD}/bfd/.libs/libbfd.a")
    set(3P_BFD_H "${3P_BINUTILS_BUILD}/bfd/bfd.h")
    set(3P_LIBSFRAME_A "${3P_BINUTILS_BUILD}/libsframe/.libs/libsframe.a")
    set(3P_ZLIB_A "${3P_BINUTILS_BUILD}/zlib/libz.a")

    if(EXISTS "${3P_LIBIBERTY_A}")
        message(STATUS "    libiberty.a: Found (pre-built)")
        set(3P_LIBIBERTY_FOUND TRUE)
        set(3P_LIBIBERTY_LIBRARY "${3P_LIBIBERTY_A}")
        set(3P_LIBIBERTY_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/binutils/include")
        # Export to parent scope
        set(3P_LIBIBERTY_FOUND TRUE PARENT_SCOPE)
        set(3P_LIBIBERTY_LIBRARY "${3P_LIBIBERTY_A}" PARENT_SCOPE)
        set(3P_LIBIBERTY_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/binutils/include" PARENT_SCOPE)
    else()
        message(STATUS "    libiberty.a: Not built (run ./3p/build_3p.sh)")
    endif()

    if(EXISTS "${3P_BFD_A}" AND EXISTS "${3P_BFD_H}")
        message(STATUS "    libbfd.a: Found (pre-built)")
        set(3P_BFD_FOUND TRUE)
        set(3P_BFD_LIBRARY "${3P_BFD_A}")
        set(3P_BFD_INCLUDE "${3P_BINUTILS_BUILD}/bfd")
        # Export to parent scope
        set(3P_BFD_FOUND TRUE PARENT_SCOPE)
        set(3P_BFD_LIBRARY "${3P_BFD_A}" PARENT_SCOPE)
        set(3P_BFD_INCLUDE "${3P_BINUTILS_BUILD}/bfd" PARENT_SCOPE)
        # Also export libsframe and zlib paths for linking
        if(EXISTS "${3P_LIBSFRAME_A}")
            message(STATUS "    libsframe.a: Found (pre-built)")
            set(3P_SFRAME_LIBRARY "${3P_LIBSFRAME_A}" PARENT_SCOPE)
        endif()
        if(EXISTS "${3P_ZLIB_A}")
            message(STATUS "    libz.a: Found (pre-built)")
            set(3P_ZLIB_LIBRARY "${3P_ZLIB_A}" PARENT_SCOPE)
        endif()
    else()
        message(STATUS "    libbfd.a: Not built (run ./3p/build_3p.sh)")
    endif()

    if(3P_LIBIBERTY_FOUND AND 3P_BFD_FOUND)
        set(_3P_LIBS_AVAILABLE TRUE)
    endif()
endif()

if(USE_3P_LIBS AND NOT _3P_LIBS_AVAILABLE)
    message(STATUS "")
    message(STATUS "USE_3P_LIBS enabled but libraries not built.")
    message(STATUS "Please build them first:")
    message(STATUS "  cd ${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "  ./build_3p.sh")
    message(STATUS "")
endif()
