# JAX Exchange Correlation Library


``` sh
bazel run //test:tmp
```

How to run:
```sh
bazel --output_user_root=$OUTPUT_USER_ROOT build --action_env=PATH=$PATH:$MAPLE_PATH //maple2jax:hyb_name_jax_xc_build
rm -f maple2jax/jax_xc.BUILD maple2jax/hyb_name
cp bazel-bin/maple2jax/jax_xc_build maple2jax/jax_xc.BUILD
cp bazel-bin/maple2jax/hyb_name maple2jax/hyb_name
bazel --output_user_root=$OUTPUT_USER_ROOT run --action_env=PATH=$PATH:$MAPLE_PATH //test:test_jax_xc
```
