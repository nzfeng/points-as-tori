import imgui
import psutil
import pyglet
import pyvista


def test_demo_dependencies_import() -> None:
	assert imgui.__version__
	assert psutil.__version__
	assert pyglet.version
	assert pyvista.__version__
