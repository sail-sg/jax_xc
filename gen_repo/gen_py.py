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
from jax_xc.utils import dict_to_namedtuple

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


def check_deorbitalize(p):
  if len(p.func_aux) != 2:
    raise ValueError(
      f"{p.name} is a DEORBITALIZE functional, "
      "it needs to have exactly two aux functionals"
    )
  for fn_p in p.func_aux:
    if fn_p.type != "mgga":
      raise ValueError(
        f"{p.name} is a DEORBITALIZE functional, "
        "it must be made of mgga functionals"
      )


def main(_):
  functionals = []
  functional_numbers = pylibxc.util.xc_available_functional_numbers()
  for number in functional_numbers:
    func = pylibxc.LibXCFunctional(number, 1)
    # get ext_params and their descriptions
    ext_params = get_ext_params(func)
    ext_params_descriptions = func.get_ext_param_descriptions()
    # get the bibtex information
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
    p = dict_to_namedtuple(p, "P")

    # check if the functional is valid
    if p.maple_name == "DEORBITALIZE":
      check_deorbitalize(p)

    functionals.append((p, ext_params, ext_params_descriptions, info))

  with open(FLAGS.template, "rt", encoding="utf8") as f:
    py_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    py_code = py_template.render(functionals=functionals, zip=zip)
    with open(FLAGS.output, "wt", encoding="utf8") as out:
      out.write(py_code)


if __name__ == "__main__":
  app.run(main)
