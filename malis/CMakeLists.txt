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
add_library(malis_loss SHARED malis_loss.cpp)
target_link_libraries(malis_loss ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY})
IF(WIN32)
    SET_TARGET_PROPERTIES(malis_loss PROPERTIES OUTPUT_NAME "malis_loss" PREFIX "" SUFFIX  ".pyd")
ELSEIF(MACOSX)
    SET_TARGET_PROPERTIES(malis_loss PROPERTIES OUTPUT_NAME "malis_loss" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(malis_loss PROPERTIES OUTPUT_NAME "malis_loss" PREFIX "")
ENDIF()
