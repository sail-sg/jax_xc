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

from absl import flags, app, logging
import os
from jinja2 import Template

FLAGS = flags.FLAGS
flags.DEFINE_string("include", None, "include maple file")
flags.DEFINE_string("output", None, "output file")
flags.DEFINE_string("template", None, "template file")
flags.DEFINE_boolean("pol", False, "if polarized")


def main(_):
  name = os.path.basename(FLAGS.include)
  if name.startswith("lda") or name.startswith("hyb_lda"):
    type = "lda"
  elif name.startswith("gga") or name.startswith("hyb_gga"):
    type = "gga"
  elif name.startswith("mgga") or name.startswith("hyb_mgga"):
    type = "mgga"
  with open(FLAGS.template, "r") as f:
    mpl_template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    logging.error(FLAGS.include)
    mpl_code = mpl_template.render(
      include=FLAGS.include,
      type=type,
      polarized=FLAGS.pol,
    )
    with open(FLAGS.output, "w") as out:
      out.write(mpl_code)


if __name__ == "__main__":
  app.run(main)
