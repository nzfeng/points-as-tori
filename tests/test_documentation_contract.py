from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FORBIDDEN = (
	'pip install',
	'Homebrew',
	'/opt/homebrew',
	'/usr/local',
	'python3.11 -m venv',
	'pyenv',
)


@pytest.mark.parametrize('forbidden', FORBIDDEN)
def test_active_project_surfaces_use_only_pixi(forbidden: str) -> None:
	surfaces = (
		ROOT / 'README.md',
		ROOT / 'CMakeLists.txt',
		ROOT / 'demo' / 'demo_3d.py',
		ROOT / 'pyproject.toml',
	)
	occurrences = [path for path in surfaces if forbidden.lower() in path.read_text().lower()]
	assert not occurrences, f'{forbidden!r} remains in {occurrences}'


def test_readme_names_reproducible_entry_points() -> None:
	readme = (ROOT / 'README.md').read_text()
	for command in (
		'pixi install --locked',
		'pixi run test',
		'pixi run build',
		'pixi run -e demo demo',
		'pixi run -e training training-contract',
	):
		assert command in readme
	assert '`osx-arm64`' in readme
