import jax
import jax.numpy as jnp

arr_1 = jnp.array(
    [[1, 2],
    [3, 4],
    [5, 6],
    [7, 8]]
)
arr_2 = jnp.array(
    [[9, 10],
    [11, 12],
    [13, 14],
    [15, 16]]
)
arr_1_ix = jnp.array([0, 2, 2, 1])
arr_2_ix = jnp.array([1, 1, 0, 2])

mask_1 = jnp.zeros_like(arr_1, dtype=bool)
mask_1 = mask_1.at[arr_1_ix == 0].set(True)
print(arr_1[mask_1])