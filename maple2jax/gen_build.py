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

import os
from absl import flags, app
from jinja2 import Template

FLAGS = flags.FLAGS
flags.DEFINE_string("output", None, "output file")
flags.DEFINE_string("impl_output", None, "impl output file")
flags.DEFINE_string("build_template", None, "build template file")
flags.DEFINE_string("impl_build_template", None, "impl build template file")


def main(_):
  py_files, names_and_maple_files = [], []
  for folder in [
    "lda_exc", "gga_exc", "mgga_exc", "lda_vxc", "gga_vxc", "mgga_vxc"
  ]:
    for src in os.listdir(f"maple/{folder}"):
      if src.endswith(".mpl"):
        maple_file = folder + "/" + src
        name = src.split(".")[0]
        py_files.append(f'\":{name}_gen_py\",')
        names_and_maple_files.append((name, maple_file))
  py_files = "\n".join(py_files)

  with open(FLAGS.build_template, "r") as f:
    build_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    build_code = build_template.render(
      py_files=py_files,
      names_and_maple_files=names_and_maple_files,
    )
    with open(FLAGS.output, "w") as out:
      out.write(build_code)

  with open(FLAGS.impl_build_template, "r") as f:
    impl_build_template = Template(
      f.read(), trim_blocks=True, lstrip_blocks=True
    )
    impl_build_code = impl_build_template.render(
      py_files=py_files,
      names_and_maple_files=names_and_maple_files,
    )
    with open(FLAGS.impl_output, "w") as out:
      out.write(impl_build_code)


if __name__ == "__main__":
  app.run(main)
