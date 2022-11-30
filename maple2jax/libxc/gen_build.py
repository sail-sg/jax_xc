#!/usr/bin/env python3
import os
from absl import flags, app
from jinja2 import Template
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string("src", None, "src dir of libxc")
flags.DEFINE_string("template", None, "input build template file")
flags.DEFINE_string("build", None, "output build file")


def main(_):
  c_file_basenames = []
  for c_file in glob.glob(FLAGS.src + "/*.c"):
    basename = os.path.basename(c_file)
    if basename.startswith("funcs_") or basename.startswith("work_"):
      continue
    else:
      c_file_basenames.append(basename)

  with open(FLAGS.template, "r") as f:
    template = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
    build = template.render(c_file_basenames=c_file_basenames)
    with open(FLAGS.build, "w") as out:
      out.write(build)


if __name__ == "__main__":
  app.run(main)