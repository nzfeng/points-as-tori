from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

HASH_BLOCK_BYTES = 1024 * 1024


class WheelContractError(RuntimeError):
	"""Built wheel violates the package or native-linkage contract."""


def newest_wheel(directory: Path) -> Path:
	wheels = sorted(
		directory.glob('pointsastori-*.whl'),
		key=lambda path: path.stat().st_mtime_ns,
	)
	if not wheels:
		raise WheelContractError(f'no pointsastori wheel in {directory}')
	return wheels[-1]


def inspect_members(wheel: Path) -> str:
	with ZipFile(wheel) as archive:
		names = archive.namelist()
	if 'pointsastori/__init__.py' not in names:
		raise WheelContractError('wheel lacks pointsastori Python sources')
	native = [name for name in names if name.startswith('pat_bindings') and name.endswith('.so')]
	if len(native) != 1:
		raise WheelContractError(f'wheel native module count is {len(native)}, expected 1')
	return native[0]


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b''):
			digest.update(block)
	return digest.hexdigest()


def linkage_command(target: Path) -> tuple[str, ...]:
	if sys.platform == 'darwin':
		return 'otool', '-L', str(target)
	if sys.platform.startswith('linux'):
		return 'ldd', str(target)
	raise WheelContractError(f'unsupported wheel verification platform: {sys.platform}')


def expected_openmp_runtime() -> str:
	if sys.platform == 'darwin':
		return 'libomp'
	if sys.platform.startswith('linux'):
		return 'libgomp'
	raise WheelContractError(f'unsupported wheel verification platform: {sys.platform}')


def verify_linkage(wheel: Path, native_member: str) -> str:
	with TemporaryDirectory() as directory:
		target = Path(directory) / Path(native_member).name
		with ZipFile(wheel) as archive:
			target.write_bytes(archive.read(native_member))
		result = subprocess.run(
			linkage_command(target),
			check=True,
			capture_output=True,
			text=True,
		)
	linkage = result.stdout
	if '/opt/homebrew' in linkage or '/usr/local' in linkage:
		raise WheelContractError(f'wheel leaks machine-local linkage:\n{linkage}')
	if expected_openmp_runtime() not in linkage:
		raise WheelContractError(f'wheel lacks OpenMP linkage:\n{linkage}')
	return linkage


def main() -> None:
	wheel = newest_wheel(Path('export/wheels'))
	native_member = inspect_members(wheel)
	linkage = verify_linkage(wheel, native_member)
	digest = sha256_file(wheel)
	print(linkage, end='')
	print(f'{digest}  {wheel}')


if __name__ == '__main__':
	try:
		main()
	except WheelContractError as error:
		print(f'wheel contract failed: {error}', file=sys.stderr)
		raise SystemExit(1) from error
