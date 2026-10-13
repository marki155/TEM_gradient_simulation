import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "temsim"
author = "Marco Kögel"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx.ext.viewcode"]
autodoc_member_order = "bysource"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
html_theme = "alabaster"
numfig = True
