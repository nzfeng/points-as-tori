#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void bind_shape_3d(nb::module_& m);
void bind_torus_distance(nb::module_& m);
void bind_bvh(nb::module_& m);
