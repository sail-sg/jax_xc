#!/usr/bin/env python3
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
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string("maple", None, "src dir of maple files")
flags.DEFINE_string("template", None, "input build template file")
flags.DEFINE_string("build", None, "output build file")


def main(_):
  names_and_maple_files = []
  for folder in [
    "lda_exc", "gga_exc", "mgga_exc", "lda_vxc", "gga_vxc", "mgga_vxc"
  ]:
    for mpl in glob.glob(FLAGS.maple + "/" + folder + "/*.mpl"):
      name = os.path.basename(mpl).split(".")[0]
      maple_file = f"{folder}/{name}.mpl"
      names_and_maple_files.append((name, maple_file))

  with open(FLAGS.template, "r") as f:
    template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    build = template.render(names_and_maple_files=names_and_maple_files)
    with open(FLAGS.build, "w") as out:
      out.write(build)


if __name__ == "__main__":
  app.run(main)
