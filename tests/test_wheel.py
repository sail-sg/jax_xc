#!/usr/bin/env python3
import jax
import jax_xc
import autofd.operators as o
from autofd import function
import jax.numpy as jnp
from jaxtyping import Array, Float32


@function
def rho(r: Float32[Array, "3"]) -> Float32[Array, ""]:
  """Electron number density. We take gaussian as an example.

  A function that takes a real coordinate, and returns a scalar
  indicating the number density of electron at coordinate r.

  Args:
    r: a 3D coordinate.

  Returns:
    rho: If it is unpolarized, it is a scalar.
      If it is polarized, it is a array of shape (2,).
  """
  return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=1))


# create a density functional
gga_x_pbe = jax_xc.experimental.gga_x_pbe
epsilon_xc = gga_x_pbe(rho)

# a grid point in 3D
r = jnp.array([0.1, 0.2, 0.3])

# pass rho and r to the functional to compute epsilon_xc (energy density) at r.
# corresponding to the 'zk' in libxc
print(f"The function signature of epsilon_xc is {epsilon_xc}")

energy_density = epsilon_xc(r)
print(f"epsilon_xc(r) = {energy_density}")

vxc = jax.grad(lambda rho: o.integrate(rho * gga_x_pbe(rho)))(rho)
print(f"The function signature of vxc is {vxc}")
print(vxc(r))
