#
# (c) Bernhard X. Kausler, 2010
# (c) Thorben Kröger, 2010
#
# This module finds an installed Vigra package.
#
# It sets the following variables:
#  VIGRA_FOUND              - Set to false, or undefined, if vigra isn't found.
#  VIGRA_INCLUDE_DIR        - Vigra include directory.
#  VIGRA_IMPEX_LIBRARY      - Vigra's impex library
#  VIGRA_IMPEX_LIBRARY_DIR  - path to Vigra impex library
#  VIGRA_NUMPY_CORE_LIBRARY - Vigra's vigranumpycore library

# configVersion.hxx only present, after build of Vigra
FIND_PATH(VIGRA_INCLUDE_DIR vigra/configVersion.hxx PATHS $ENV{VIGRA_ROOT}/include ENV CPLUS_INCLUDE_PATH)

# handle the QUIETLY and REQUIRED arguments and set VIGRA_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VIGRA DEFAULT_MSG VIGRA_INCLUDE_DIR)
IF(VIGRA_FOUND)
    IF (NOT Vigra_FIND_QUIETLY)
      MESSAGE(STATUS "  > includes:      ${VIGRA_INCLUDE_DIR}")
    ENDIF()
ENDIF()

MARK_AS_ADVANCED( VIGRA_INCLUDE_DIR )
