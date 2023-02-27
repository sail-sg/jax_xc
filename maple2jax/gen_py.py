# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

import re
import ctypes
from absl import flags, app
from jinja2 import Template

import jax_xc.libxc as pylibxc
from jax_xc.libxc import libxc

FLAGS = flags.FLAGS
flags.DEFINE_string("output", None, "output py file")
flags.DEFINE_string("template", None, "template file")
flags.DEFINE_string("pointwise_output", None, "pointwise output py file")
flags.DEFINE_string("pointwise_template", None, "pointwise template file")


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
    ext_params_descriptions = func.get_ext_param_descriptions()

    bibtexes = func.get_bibtex()
    # find url = \{(.*?)\} in bibtex, adding |$ to set default to ""
    urls = [re.findall(r"url = \{(.*?)\}|$", bibtex)[0] for bibtex in bibtexes]
    # undo libxc escaping
    urls = [re.sub(r"\\(.)", r"\1", url) for url in urls]
    # escape < and > for rst url format
    urls = [url.replace("<", r"\<").replace(">", r"\>") for url in urls]
    dois = func.get_doi()
    refs = func.get_references()
    # merge urls, dois, refs into a list of tuples
    info = [
      (url.strip(), doi.strip(), ref.strip())
      for url, doi, ref in zip(urls, dois, refs)
    ]

    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    name = p["name"]
    maple_name = p["maple_name"]
    # special cases
    if name == 'mgga_x_2d_prhg07_prp10':
      maple_name = 'mgga_x_2d_prp10'
    if name == 'hyb_mgga_xc_b98':
      maple_name = 'mgga_xc_b98'

    aux_info = zip([fn_aux['name'] for fn_aux in p['func_aux']],
                   p['mix_coef']) if 'func_aux' in p else []

    functionals.append(
      (name, ext_params, maple_name, ext_params_descriptions, info, aux_info)
    )

  with open(FLAGS.template, "r") as f:
    py_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    py_code = py_template.render(functionals=functionals, zip=zip)
    with open(FLAGS.output, "w") as out:
      out.write(py_code)

  with open(FLAGS.pointwise_template, "r") as f:
    py_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    py_code = py_template.render(functionals=functionals, zip=zip)
    with open(FLAGS.pointwise_output, "w") as out:
      out.write(py_code)


if __name__ == "__main__":
  app.run(main)
