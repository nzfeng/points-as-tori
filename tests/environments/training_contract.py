import matplotlib
import noise
import py7zr
import pyfqmr
import pyvista
import scipy
import thingi10k
import trimesh
from stl import mesh


def test_training_dependencies_import() -> None:
	assert matplotlib is not None
	assert noise.snoise2(0.0, 0.0) == 0.0
	assert py7zr is not None
	assert pyfqmr is not None
	assert pyvista is not None
	assert scipy is not None
	assert thingi10k is not None
	assert trimesh is not None
	assert mesh.Mesh is not None
