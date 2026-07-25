cmake_minimum_required(VERSION 3.15)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(pat_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -fPIC")

find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

find_package(OpenMP REQUIRED)

if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "Pixi environment required: CONDA_PREFIX is not set")
endif()

set(PIXI_PREFIX "$ENV{CONDA_PREFIX}")
set(PIXI_INCLUDE_DIR "${PIXI_PREFIX}/include")
set(PIXI_LIBRARY_DIR "${PIXI_PREFIX}/lib")
set(PIXI_NUMERIC_FILES
    "${PIXI_INCLUDE_DIR}/gmp.h"
    "${PIXI_INCLUDE_DIR}/gmpxx.h"
    "${PIXI_INCLUDE_DIR}/mpfr.h"
    "${PIXI_LIBRARY_DIR}/libgmp.dylib"
    "${PIXI_LIBRARY_DIR}/libgmpxx.dylib"
    "${PIXI_LIBRARY_DIR}/libmpfr.dylib"
)
foreach(PIXI_NUMERIC_FILE IN LISTS PIXI_NUMERIC_FILES)
    if(NOT EXISTS "${PIXI_NUMERIC_FILE}")
        message(FATAL_ERROR "Pixi numeric dependency missing: ${PIXI_NUMERIC_FILE}")
    endif()
endforeach()

set(GMP_INCLUDE_DIR "${PIXI_INCLUDE_DIR}" CACHE PATH "Pixi GMP headers" FORCE)
set(GMP_LIBRARY_DEBUG "${PIXI_LIBRARY_DIR}/libgmp.dylib" CACHE FILEPATH "Pixi debug GMP library" FORCE)
set(GMP_LIBRARY_RELEASE "${PIXI_LIBRARY_DIR}/libgmp.dylib" CACHE FILEPATH "Pixi release GMP library" FORCE)
set(GMPXX_INCLUDE_DIR "${PIXI_INCLUDE_DIR}" CACHE PATH "Pixi GMPXX headers" FORCE)
set(GMPXX_LIBRARIES "${PIXI_LIBRARY_DIR}/libgmpxx.dylib" CACHE FILEPATH "Pixi GMPXX library" FORCE)
set(MPFR_INCLUDE_DIR "${PIXI_INCLUDE_DIR}" CACHE PATH "Pixi MPFR headers" FORCE)
set(MPFR_LIBRARIES "${PIXI_LIBRARY_DIR}/libmpfr.dylib" CACHE FILEPATH "Pixi MPFR library" FORCE)
set(MPFR_LIBRARIES_DIR "${PIXI_LIBRARY_DIR}" CACHE PATH "Pixi MPFR library directory" FORCE)

set(LIBIGL_COPYLEFT_CGAL ON CACHE BOOL "Use CGAL" FORCE)

# https://github.com/broccoli-97/libigl-example-project/blob/main/CMakeLists.txt
find_package(CGAL QUIET COMPONENTS Core)
option(LIBIGL_WITH_CGAL            "Use CGAL"          "${CGAL_FOUND}")
option(LIBIGL_COPYLEFT_CGAL       "Use CGAL"           ON)

if(LIBIGL_WITH_CGAL) # Do not remove or move this block, cgal strange build system fails without it
  find_package(CGAL REQUIRED COMPONENTS Core)
  set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "CGAL's CMAKE Setup is super annoying ")
  message(STATUS "  CGAL_USE_FILE = ${CGAL_USE_FILE}")
  #include(${CGAL_USE_FILE})
endif()

# Dependencies (as git submodules)
add_subdirectory(deps/fcpw)
add_subdirectory(deps/nanobind)
add_subdirectory(deps/nanoflann)
add_subdirectory(deps/libigl)

# The Python extension module
nanobind_add_module(
    pat_bindings
    STABLE_ABI
    NB_STATIC
    src/cpp/core.cpp
    src/cpp/torus_distance.cpp
    src/cpp/shape_3d.cpp
    src/cpp/shape_2d.cpp
    src/cpp/bvh.cpp
)

target_include_directories(pat_bindings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)
target_include_directories(pat_bindings PUBLIC
    ${FCPW_ENOKI_INCLUDES}
    ${FCPW_EIGEN_INCLUDES}
)
target_include_directories(pat_bindings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include
)

target_link_libraries(pat_bindings PRIVATE
    nanoflann::nanoflann
    fcpw
    igl::core
    igl_copyleft::cgal
    OpenMP::OpenMP_CXX
)

install(TARGETS pat_bindings LIBRARY DESTINATION .)
