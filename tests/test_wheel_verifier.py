from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.verify_wheel import WheelContractError, inspect_members, sha256_file


def _wheel(path: Path, members: tuple[str, ...]) -> Path:
	with ZipFile(path, 'w') as archive:
		for member in members:
			archive.writestr(member, b'content')
	return path


def test_wheel_requires_python_and_native_members(tmp_path: Path) -> None:
	wheel = _wheel(
		tmp_path / 'pointsastori.whl',
		('pointsastori/__init__.py', 'pat_bindings.abi3.so'),
	)
	assert inspect_members(wheel) == 'pat_bindings.abi3.so'


def test_wheel_rejects_missing_native_member(tmp_path: Path) -> None:
	wheel = _wheel(
		tmp_path / 'pointsastori.whl',
		('pointsastori/__init__.py',),
	)
	with pytest.raises(WheelContractError, match='native'):
		inspect_members(wheel)


def test_sha256_is_stable(tmp_path: Path) -> None:
	artifact = tmp_path / 'artifact'
	artifact.write_bytes(b'points-as-tori')
	assert sha256_file(artifact) == '6e58ac94a24ae9c1d356f34a8a03419786415c1b54efa7c911c5a415371cff31'
