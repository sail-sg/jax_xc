# JAX Exchange Correlation Library

This library contains direct translations of exchange correlation functionals in [libxc](https://tddft.org/programs/libxc/)
to [jax](https://github.com/google/jax). The core calculations in libxc are implemented in [maple](https://www.maplesoft.com/).
This gives us the opportunity to translate them directly into python with the help of
[CodeGeneration](https://www.maplesoft.com/support/help/maple/view.aspx?path=CodeGeneration%2fPython).

## Usage

### Installation

``` sh
pip install jax-xc
```

### Invoking the Functionals

#### LDA and GGA

Unlike `libxc` which takes pre-computed densities and their derivative at certain coordinates.
In `jax_xc`, the API is designed to directly take a density function.

``` python
import jax_xc

def rho(r):
  """Electron number density.

  A function that takes a real coordinate, and returns a scalar
  indicating the number density of electron at coordinate r.

  Args:
    r: a 3D coordinate.
  Returns:
    rho: If it is unpolarized, it is a scalar.
         If it is polarized, it is a array of shape (2,).
  """
  pass

exc = jax_xc.gga_xc_pbe1w(rho=rho, polarized=False)


# Numerical integral with grids and their corresponding weights

rho_times_exc = lambda w, r: w * rho(r) * exc(r)
Exc = jnp.sum(vmap(rho_times_exc)(weights, grids))
```

#### mGGA

Unlike LDA and GGA that only depends on the density function,
mGGA functionals also depend on the molecular orbitals.

``` python
import jax_xc

def mo(r):
  """Molecular orbital.

  A function that takes a real coordinate, and returns the value of
  molecular orbital at this coordinate.

  Args:
    r: a 3D coordinate.
  Returns:
    mo: If it is unpolarized, it is a array of shape (N,).
        If it is polarized, it is a array of shape (N, 2).
  """
  pass

exc = jax_xc.mgga_xc_cc06(rho=rho, polarized=polarized, mo=mo)

# perform numerical integral like the example in LDA and GGA
```

#### Hybrid Functionals

## Numerical Correctness

## Performance Benchmark

## Caveates

The following functionals from `libxc` are not available in `jax_xc` because some functions are not available in `jax`.

## Building from Source Code


Modify the `.env.example` to fill in your envrionment variables, then rename it to `.env`. Then run `source .env` to load them into your shell.

- `OUTPUT_USER_ROOT`: The path to the bazel cache. This is where the bazel cache will be stored. This is useful if you are building on a shared filesystem.

- `MAPLE_PATH`: The path to the maple binary.

- `TMP_INSTALL_PATH`: The path to a temporary directory where the wheel will be installed. This is useful if you are building on a shared filesystem.


How to build.

``` sh
bazel --output_user_root=$OUTPUT_USER_ROOT build --action_env=PATH=$PATH:$MAPLE_PATH @maple2jax//:jax_xc_wheel
```

How to generate and serve documentation.

``` sh
bazel --output_user_root=$OUTPUT_USER_ROOT build --action_env=PATH=$PATH:$MAPLE_PATH @maple2jax//:jax_xc_wheel
pip install --force-reinstall --upgrade -t $TMP_INSTALL_PATH bazel-bin/external/maple2jax/jax_xc-0.0.1-py3-none-any.whl
cd docs
export PYTHONPATH=$TMP_INSTALL_PATH
make html
sphinx-serve
```
