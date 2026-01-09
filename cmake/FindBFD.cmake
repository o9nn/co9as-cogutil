# Check for 3p pre-built BFD first (when USE_3P_LIBS is enabled)
if(USE_3P_LIBS AND 3P_BFD_FOUND)
    message(STATUS "Using 3p libbfd: ${3P_BFD_LIBRARY}")
    set(BFD_LIBRARY "${3P_BFD_LIBRARY}")
    set(BFD_INCLUDE_DIR "${3P_BFD_INCLUDE}")
    set(BFD_FOUND TRUE)
    set(HAVE_BFD_H TRUE)
else()
    # find_library( DL_LIBRARY 	NAMES dl	PATH /usr/lib /usr/lib64 )
    find_library( BFD_LIBRARY	NAMES bfd	PATH /usr/lib /usr/lib64 )

    # # Check to see if bfd cn compile and link ...
    # include(CheckCSourceCompiles)
    # check_c_source_compiles(
    #   "#include <bfd.h>
    #   int main(void) {
    #   return 0;
    #   }" BFD_WORKS)

    #if (DL_LIBRARY AND BFD_LIBRARY AND BFD_WORKS)
    #	set( BFD_FOUND TRUE )
    #endif (DL_LIBRARY AND BFD_LIBRARY AND BFD_WORKS)

    INCLUDE (CheckIncludeFiles)
    CHECK_INCLUDE_FILES (bfd.h HAVE_BFD_H)

    if (BFD_LIBRARY AND HAVE_BFD_H)
        set( BFD_FOUND TRUE )
    endif (BFD_LIBRARY AND HAVE_BFD_H)

    if ( BFD_FOUND )
        message( STATUS "Found libbfd: ${BFD_LIBRARY}")
    else ( BFD_FOUND )
        message( STATUS "BFD not found")
    endif ( BFD_FOUND )
endif()
