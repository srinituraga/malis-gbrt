cmake_minimum_required(VERSION 2.8)
PROJECT(malis)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Vigra REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)
find_package(PythonLibs REQUIRED)
#find_package(Numpy REQUIRED)

include_directories(
    ${VIGRA_INCLUDE_DIR}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_PATH}
)
add_library(pymalis SHARED malis_loss_py.cpp)
target_link_libraries(pymalis ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY})
IF(WIN32)
    SET_TARGET_PROPERTIES(pymalis PROPERTIES OUTPUT_NAME "pymalis" PREFIX "" SUFFIX  ".pyd")
ELSEIF(MACOSX)
    SET_TARGET_PROPERTIES(pymalis PROPERTIES OUTPUT_NAME "pymalis" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(pymalis PROPERTIES OUTPUT_NAME "pymalis" PREFIX "")
ENDIF()

add_library(tree SHARED tree.cpp)
target_link_libraries(tree ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY})
IF(WIN32)
    SET_TARGET_PROPERTIES(tree PROPERTIES OUTPUT_NAME "tree" PREFIX "" SUFFIX  ".pyd")
ELSEIF(MACOSX)
    SET_TARGET_PROPERTIES(tree PROPERTIES OUTPUT_NAME "tree" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(tree PROPERTIES OUTPUT_NAME "tree" PREFIX "")
ENDIF()

