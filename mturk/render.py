from jinja2 import Template

with open("task_template.html", "r") as f:
    t = Template(f.read())
out = t.render()
with open("task_output.html", "w") as f2:
    f2.write(out)
