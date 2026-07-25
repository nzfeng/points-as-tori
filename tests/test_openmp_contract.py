import os
import subprocess
import sys
import textwrap

import pytest

PROBE = textwrap.dedent(
	"""
    import os
    import sys

    import numpy as np

    order = sys.argv[1]
    if order == "jax-first":
        import jax.numpy as jnp
        import pat_bindings
    else:
        import pat_bindings
        import jax.numpy as jnp

    assert "KMP_DUPLICATE_LIB_OK" not in os.environ
    matrix = jnp.arange(512 * 512, dtype=jnp.float32).reshape(512, 512)
    product = (matrix @ matrix.T).block_until_ready()
    assert bool(jnp.isfinite(product).all())

    circle = pat_bindings.Circle2D(1.0)
    queries = np.zeros((2, 20_000), dtype=np.float64)
    distances = circle.evaluate_batch(queries)
    assert distances.shape == (20_000,)
    assert np.isfinite(distances).all()
    assert np.all(distances == -1.0)
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ
    print("CLEAN")
    """
)


@pytest.mark.parametrize('order', ['jax-first', 'pat-first'])
def test_jax_and_pat_share_the_process_without_omp_escape_hatch(
	order: str,
) -> None:
	env = dict(os.environ)
	env.pop('KMP_DUPLICATE_LIB_OK', None)
	result = subprocess.run(
		[sys.executable, '-c', PROBE, order],
		capture_output=True,
		check=False,
		text=True,
		env=env,
	)
	assert result.returncode == 0, f'order={order}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
	assert 'CLEAN' in result.stdout
