# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

from absl import flags, app
import os
import regex as re
from jinja2 import Template
import jax.numpy as jnp

FLAGS = flags.FLAGS
flags.DEFINE_string("pol", None, "input pol code file")
flags.DEFINE_string("unpol", None, "input unpol code file")
flags.DEFINE_string("output", None, "output py file")
flags.DEFINE_string("template", None, "template file")


def post_process(py_code):
  replace_rules = (
    ("_a_", "."),
    # convert constants like 0.225000000e-1 to 0.225e-1
    (r"0+e", r"e"),
    # remove the e0 and e00
    (r"([\d\.]+)e0+", r"\1"),
    (r"_(\d+)_", r"[\1]"),
    # convert numerical value of pi to constant
    (r"0.31415926535897932385e1", r"jnp.pi"),
    (r"math.erf", r"jax.lax.erf"),
    # convert parenthesis ** (0.1e1 / 0.3e1) to cbrt using recursive regex
    (
      r"(.*)(\((?:[^)(]+|(?2))*+\)) \*\* \(0.1e1 \/ 0.3e1\)(.*)",
      r"\1jnp.cbrt\2\3"
    ),
    # convert variable ** (0.1e1 / 0.3e1) to cbrt, order matters
    (r"(.*) (.*) \*\* \(0.1e1 \/ 0.3e1\)(.*)", r"\1 jnp.cbrt(\2)\3"),
    ("scipy.special.i0", "jax.scipy.special.i0"),
    ("scipy.special.lambertw", "lambertw"),
    ("math", "jnp"),
    ("atan", "arctan"),
    ("asinh", "arcsinh"),
    ("abs", "jnp.abs"),
    ("DBL_EPSILON", f"{jnp.finfo(float).eps}"),
    # for Python
    ("lambda", "lambda_"),
  )
  for old, new in replace_rules:
    py_code = re.sub(old, new, py_code)
  return py_code


def main(_):
  name = os.path.basename(FLAGS.output).split(".")[0]
  if name.startswith("lda") or name.startswith("hyb_lda"):
    type = "lda"
  elif name.startswith("gga") or name.startswith("hyb_gga"):
    type = "gga"
  elif name.startswith("mgga") or name.startswith("hyb_mgga"):
    type = "mgga"

  with open(FLAGS.pol, "r") as pol:
    pol_code = pol.read()
    pol_code = post_process(pol_code)
  with open(FLAGS.unpol, "r") as unpol:
    unpol_code = unpol.read()
    unpol_code = post_process(unpol_code)

  with open(FLAGS.template, "r") as f:
    py_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    py_code = py_template.render(
      name=name,
      type=type,
      pol_code=pol_code.strip(),
      unpol_code=unpol_code.strip(),
    )
    with open(FLAGS.output, "w") as out:
      out.write(py_code)


if __name__ == "__main__":
  app.run(main)
