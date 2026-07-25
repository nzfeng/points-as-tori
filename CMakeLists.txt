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
