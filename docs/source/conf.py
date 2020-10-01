# -*- coding: utf-8 -*-

import os
import subprocess

import sphinx_rtd_theme

read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
if read_the_docs_build:
    subprocess.call('cd ..; doxygen', shell=True)

# -- Project information -----------------------------------------------------

project = 'GTN'
copyright = '2020, GTN Contributors'
author = 'GTN Contributors'
version = '0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'breathe'
]

intersphinx_mapping = {
    'https://docs.python.org/3': None,
    'https://numpy.org/doc/stable/': None,
    'https://pytorch.org/docs/stable/' : None,
}

breathe_projects = {"gtn" : "../build/xml"}
breathe_default_project = "gtn"
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
highlight_language = 'c++'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

def setup(app):
    app.add_css_file("css/styles.css")

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'gtndoc'
