import numpy as np
import pat_bindings

import pointsastori


def test_python_package_and_native_module_import() -> None:
	assert pointsastori.PointsAsTori is not None
	assert pat_bindings.__doc__ == 'Points as Tori C++ bindings'


def test_circle_binding_evaluates_signed_distance() -> None:
	circle = pat_bindings.Circle2D(1.0)
	distance = circle.evaluate(np.array([0.0, 0.0]))
	assert distance == -1.0
