# -*- coding: utf-8 -*-

import os
import subprocess

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'GTN'
copyright = '2020, GTN Contributors'
author = 'GTN Contributors'
version = '0.0'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = 'Python'
pygments_style = 'sphinx'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

def setup(app):
    app.add_css_file("css/styles.css")

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'gtndoc'
