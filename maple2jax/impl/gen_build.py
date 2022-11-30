#!/usr/bin/env python3

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
      maple_file = f"maple/{folder}/{name}.mpl"
      names_and_maple_files.append((name, maple_file))

  with open(FLAGS.template, "r") as f:
    template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    build = template.render(names_and_maple_files=names_and_maple_files)
    with open(FLAGS.build, "w") as out:
      out.write(build)


if __name__ == "__main__":
  app.run(main)
