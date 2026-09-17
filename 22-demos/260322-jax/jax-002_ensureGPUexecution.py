import jax.numpy as jnp
x = jnp.ones((3, 3))
y = x @ x
print(y.devices(), y)
