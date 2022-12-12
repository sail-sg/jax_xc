# JAX Exchange Correlation Library

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
