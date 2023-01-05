# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def get_additional_import(py_code):
  additional = []
  if 'Heaviside' in py_code:
    additional.append("Heaviside")
  if 'xc_E1_scaled' in py_code:
    additional.append("xc_E1_scaled")
  if 'scipy.special.lambertw' in py_code:
    additional.append("lambertw")
  if 'xc_erfcx' in py_code:
    additional.append("xc_erfcx")
  return additional


def post_process(py_code):
  replace_rules = (
    ("_a_", "."),
    # convert constants like 0.225000000e-1 to 0.225e-1
    (r"0+e", r"e"),
    # remove the e0 and e00
    (r"e0", r""),
    (r"e00", r""),
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
  with open(FLAGS.unpol, "r") as unpol:
    unpol_code = unpol.read()

  additional = get_additional_import(pol_code + unpol_code)
  additional = ", ".join(additional)

  with open(FLAGS.template, "r") as f:
    py_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    py_code = py_template.render(
      name=name,
      type=type,
      pol_code=pol_code.strip(),
      unpol_code=unpol_code.strip(),
      additional=additional,
    )
    py_code = post_process(py_code)
    with open(FLAGS.output, "w") as out:
      out.write(py_code)


if __name__ == "__main__":
  app.run(main)
