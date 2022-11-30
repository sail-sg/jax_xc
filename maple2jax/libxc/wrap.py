"""wrap the c code in libxc"""

import os
import re
from collections import defaultdict
from absl import flags, app
from jinja2 import Template

FLAGS = flags.FLAGS
flags.DEFINE_string("template", None, "template file")
flags.DEFINE_string("path", None, "source path")
flags.DEFINE_string("out", None, "out path")


def wrap_file(filename, out):
  with open(filename, "r") as f:
    content = f.read()
    # find all init function and the corresponding param struct name
    results = re.findall(
      r"(\w*?_init)\(xc_func_type \*p\).*?(?:\n|.)*?libxc_malloc\(sizeof\((\w*)",  # noqa
      content,
      re.MULTILINE,
    )

    maple2c_name = re.findall(r'#include "maple2c\/.*\/(.*).c"', content)
    work_name = re.findall(r'#include "(work_.*)\.c"', content)
    if len(maple2c_name) > 1:
      raise ValueError("more than one maple2c_src found")
    if len(work_name) > 1:
      raise ValueError("work_src should be unique")
    register_maple = (work_name[0], maple2c_name[0]) if work_name else None

    struct_to_init = defaultdict(list)
    for init, struct in results:
      struct_to_init[struct].append(init)
    structs = set([s for _, s in results])
    register_struct = []
    for s in structs:
      # find the struct definition
      struct_results = re.findall(
        r"typedef struct(?:\n|\s)*?\{((?:\n|.)*?)\}\s*?" + s,
        content,
        re.MULTILINE,
      )
      if len(struct_results) == 0:
        raise RuntimeError(
          f"Could not find struct definition for {s} in {filename}"
        )
      if len(struct_results) > 1:
        raise RuntimeError(
          f"Find more than one struct definition for {s} in {filename}, "
          "check the regex."
        )
      # remove comments in c code
      result = re.sub(r"/\*(.|[\r\n])*?\*/", "", struct_results[0])
      # remove [\d] in c array definition
      result = re.sub(r"\[\d*\]", "", result)
      groups = map(lambda s: s.strip(), result.split(";"))
      fields = []
      # in each group, find the field names and strip the type
      for g in groups:
        members = list(filter(lambda s: len(s) != 0, re.split(",| ", g)))
        if len(members) == 0:
          continue
        elif members[0] == "const":  # sometimes they add a const
          members = members[2:]
        else:
          members = members[1:]
        fields.extend(members)
      register_struct.append((s, fields, struct_to_init[s]))

    with open(FLAGS.template, "r") as f:
      t = Template(f.read(), trim_blocks=True, lstrip_blocks=True)
      content = t.render(
        filename=os.path.basename(filename),
        register_struct=register_struct,
        register_maple=register_maple,
      )
      with open(FLAGS.out, "wt") as fout:
        fout.write(content)


def main(_):
  wrap_file(FLAGS.path, FLAGS.out)


if __name__ == "__main__":
  app.run(main)
