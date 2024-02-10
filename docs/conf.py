# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "s1etad"
copyright = "2020-2023, s1etad Developers"
author = "Nuno Miranda, Antonio Valentino"

with open("../s1etad/__init__.py") as fd:
    s = fd.read()

pattern = (
    r"^__version__( )?=( )?"
    r'(?P<q>[\'"])(?P<r>(?P<v>\d+\.\d+(\.\d+)?).*)(?P=q)'
)

# The short X.Y version
version = re.search(pattern, s, re.M).group("v")

# The full version, including alpha/beta/rc tags
release = re.search(pattern, s, re.M).group("r")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosectionlabel",
    # "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.extlinks",
    # "sphinx.ext.githubpages",
    # "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    # "sphinx.ext.imgconverter",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.linkcode",  # needs_sphinx = "1.2"
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # "sphinx.ext.imgmath",
    # "sphinx.ext.jsmath",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_rtd_theme",
]

try:
    import sphinxcontrib.spelling  # noqa: F401
except ImportError:
    pass
else:
    extensions.append("sphinxcontrib.spelling")

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    # 'vcs_pageview_mode': 'blob',
}
html_context = {
    "github_url": "https://github.com/s1tools/s1-etad",
    "display_github": True,
    "github_user": "s1tools",
    "github_repo": "s1-etad",
    "github_version": "main",
    "conf_py_path": "docs",
}

# html_last_updated_fmt = ""


# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------
# autoclass_content = 'both'
autodoc_member_order = "groupwise"
# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "show-inheritance": True,
# }
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "lxml",
    "netCDF4",
    "pandas",
    "dateutil",
    "shapely",
    "simplekml",
    "osgeo",
    "matplotlib",
]


# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True
# autosummary_mock_imports = []
# autosummary_ignore_module_all = False


# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# -- Options for extlinks extension ------------------------------------------

extlinks = {
    "issue": ("https://github.com/s1tools/s1-etad/issues/%s", "gh-%s"),
}


# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
