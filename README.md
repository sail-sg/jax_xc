# JAX Exchange Correlation Library

How to build.

``` sh
bazel --output_user_root=$OUTPUT_USER_ROOT build --action_env=PATH=$PATH:$MAPLE_PATH @maple2jax//:jax_xc_wheel
```

How to generate and serve documentation.

``` sh
bazel --output_user_root=$OUTPUT_USER_ROOT build --action_env=PATH=$PATH:$MAPLE_PATH @maple2jax//:jax_xc_wheel
pip install --force-reinstall --upgrade -t /tmp/jax_xc_install bazel-bin/external/maple2jax/jax_xc-0.0.1-py3-none-any.whl
cd docs
export PYTHONPATH=/tmp/jax_xc_install
make html
sphinx-serve
```
