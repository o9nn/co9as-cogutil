# Check for 3p pre-built libiberty first (when USE_3P_LIBS is enabled)
if(USE_3P_LIBS AND 3P_LIBIBERTY_FOUND)
    message(STATUS "Using 3p libiberty: ${3P_LIBIBERTY_LIBRARY}")
    set(IBERTY_LIBRARY "${3P_LIBIBERTY_LIBRARY}")
    set(IBERTY_INCLUDE_DIR "${3P_LIBIBERTY_INCLUDE}")
    set(IBERTY_FOUND TRUE)
    set(HAVE_IBERTY_H TRUE)
else()
    find_library( IBERTY_LIBRARY	NAMES iberty	PATH /usr/lib /usr/lib64 )

    INCLUDE (CheckIncludeFiles)

    # The location of iberty.h varies according to the distro.
    # Find the right location.
    find_path(
        IBERTY_INCLUDE_DIR libiberty.h
        PATHS
            /usr/include
            /usr/local/include
            /usr/include/libiberty
            /usr/local/include/libiberty
    )

    set(CMAKE_REQUIRED_INCLUDES ${IBERTY_INCLUDE_DIR})
    CHECK_INCLUDE_FILES (libiberty.h HAVE_IBERTY_H)

    if (IBERTY_LIBRARY AND HAVE_IBERTY_H)
        set( IBERTY_FOUND TRUE )
    endif (IBERTY_LIBRARY AND HAVE_IBERTY_H)

    if ( IBERTY_FOUND )
        message( STATUS "Found libiberty: ${IBERTY_LIBRARY}")
    else ( IBERTY_FOUND )
        message( STATUS "IBERTY not found")
    endif ( IBERTY_FOUND )
endif()
