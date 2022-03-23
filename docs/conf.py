# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
sys.path.insert(0, os.path.abspath('..'))

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 's1etad'
copyright = '2020-2022, s1etad Developers'
author = 'Nuno Miranda <nuno.miranda@esa.int>'

with open('../s1etad/__init__.py') as fd:
    s = fd.read()

pattern = (
    r'^__version__( )?=( )?'
    r'(?P<q>[\'"])(?P<r>(?P<v>\d+\.\d+(\.\d+)?).*)(?P=q)'
)

# The short X.Y version
version = re.search(pattern, s, re.M).group('v')

# The full version, including alpha/beta/rc tags
release = re.search(pattern, s, re.M).group('r')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'default'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/3/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = 'groupwise'
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'lxml',
    'netCDF4',
    'pandas',
    'dateutil',
    'shapely',
    'simplekml',
    'osgeo',
    'matplotlib',
]

# -- Options for ReadTheDocs integration -------------------------------------
master_doc = 'index'
html_context = {
    "display_gitlab": True,
    "gitlab_user": "s1-etad",
    "gitlab_repo": "s1-etad",
    "gitlab_version": "master",
    "conf_py_path": "docs",
}
