#include "core.h"

NB_MODULE(pat_bindings, m) {
    m.doc() = "Points as Tori C++ bindings";

    bind_shape_2d(m);
    bind_shape_3d(m);
    bind_torus_distance(m);
    bind_bvh(m);
}
