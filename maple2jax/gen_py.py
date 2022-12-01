import re
import ctypes
from absl import flags, app
from jinja2 import Template

import jax_xc.libxc as pylibxc
from jax_xc.libxc import libxc

FLAGS = flags.FLAGS
flags.DEFINE_string("output", None, "output py file")
flags.DEFINE_string("template", None, "template file")


def post_process(param_name):
  replace_rules = (
    (r"\[(\d+)\]", r"_\1_"),
    # for Python
    ("lambda", "lambda_"),
  )
  for old, new in replace_rules:
    param_name = re.sub(old, new, param_name)
  return param_name


def get_ext_params(func):
  try:
    ext_param_names = func.get_ext_param_names()
    ext_param_names = [post_process(i) for i in ext_param_names]
    ext_param_default_values = func.get_ext_param_default_values()
    ext_params = dict(zip(ext_param_names, ext_param_default_values))
  except AttributeError:
    ext_params = {}
  return ext_params


def main(_):
  functionals = []
  functional_numbers = pylibxc.util.xc_available_functional_numbers()
  for number in functional_numbers:
    func = pylibxc.LibXCFunctional(number, 1)
    ext_params = get_ext_params(func)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    name = p["name"]
    maple_name = p["maple_name"]
    # special cases
    if name == 'mgga_x_2d_prhg07_prp10':
      maple_name = 'mgga_x_2d_prp10'
    if name == 'hyb_mgga_xc_b98':
      maple_name = 'mgga_xc_b98'

    if name.startswith("lda") or name.startswith("hyb_lda"):
      type = "lda"
    elif name.startswith("gga") or name.startswith("hyb_gga"):
      type = "gga"
    elif name.startswith("mgga") or name.startswith("hyb_mgga"):
      type = "mgga"

    functionals.append((name, type, ext_params, maple_name))

  with open(FLAGS.template, "r") as f:
    py_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    py_code = py_template.render(functionals=functionals)
    with open(FLAGS.output, "w") as out:
      out.write(py_code)


if __name__ == "__main__":
  app.run(main)
