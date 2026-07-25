from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).parents[1]


def _manifest() -> dict[str, Any]:
	return tomllib.loads((ROOT / 'pyproject.toml').read_text())


def test_pixi_owns_the_osx_arm64_build() -> None:
	pixi = _manifest()['tool']['pixi']
	assert pixi['workspace'] == {
		'channels': ['conda-forge'],
		'platforms': ['osx-arm64'],
	}
	assert pixi['system-requirements']['macos'] == '15.0'
	dependencies = pixi['dependencies']
	assert dependencies['llvm-openmp'].startswith('22.')
	assert dependencies['cgal-cpp'].startswith('6.')
	assert dependencies['libopenblas']['build'] == '*openmp*'
	assert pixi['feature']['py312']['dependencies']['python'] == '3.12.*'


def test_pixi_defines_the_core_gates() -> None:
	tasks = _manifest()['tool']['pixi']['tasks']
	assert {'test', 'lint', 'format-check', 'build', 'check'} <= tasks.keys()
	assert tasks['test']['cmd'] == 'pytest -n auto --testmon'
	assert tasks['check']['depends-on'] == [
		'lint',
		'format-check',
		'test',
		'build',
	]


def test_scikit_build_configuration_is_nested_correctly() -> None:
	manifest = _manifest()
	assert 'cmake' not in manifest
	scikit_build = manifest['tool']['scikit-build']
	assert scikit_build['logging']['level'] == 'INFO'
	assert 'sdist' in scikit_build
	assert scikit_build['strict-config'] is False


def test_cmake_requires_pixi_openmp_without_machine_paths() -> None:
	cmake = (ROOT / 'CMakeLists.txt').read_text()
	assert 'find_package(OpenMP REQUIRED)' in cmake
	assert 'OpenMP_CXX_FLAGS' not in cmake
	assert '/opt/homebrew' not in cmake
	assert '/usr/local' not in cmake
	assert 'OpenMP::OpenMP_CXX' in cmake
