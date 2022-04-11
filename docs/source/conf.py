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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import datetime

import pkg_resources

# import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "pivuq"
copyright = f"{datetime.datetime.today().year}, Lento Manickathan"
author = "Lento Manickathan"

# The full version, including alpha/beta/rc tags
release = pkg_resources.get_distribution(project).version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []
extensions = [
    # 'matplotlib.sphinxext.plot_directive', # see: https://matplotlib.org/sampledoc/extensions.html  # noqa: E501
    "sphinx.ext.autodoc",  # include documentation from docstrings
    "sphinx.ext.doctest",  # >>> examples
    "sphinx.ext.extlinks",  # for :pr:, :issue:, :commit:
    "sphinx.ext.autosectionlabel",  # use :ref:`Heading` for any heading
    "sphinx.ext.todo",  # Todo headers and todo:: directives
    "sphinx.ext.mathjax",  # LaTeX style math
    "sphinx.ext.viewcode",  # view code links
    "sphinx.ext.napoleon",  # for NumPy style docstrings
    "sphinx.ext.intersphinx",  # external links
    "sphinx.ext.autosummary",  # autosummary directive
    "sphinx_copybutton",
    "nbsphinx",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Fix duplicate class member documentation from autosummary + numpydoc
# See: https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = False

napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Python or inherited terms
    # NOTE: built-in types are automatically included
    "callable": ":py:func:`callable`",
    "sequence": ":term:`sequence`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "array-like": ":term:`array-like <array_like>`",
}
