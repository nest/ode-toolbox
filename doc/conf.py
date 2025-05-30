# -*- coding: utf-8 -*-
#
# conf.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Readthedocs configuration file
------------------------------

Use:
sphinx-build -c ../extras/help_generator -b html . _build/html

"""

import sys
import os


import pip

# pip.main(['install', 'Sphinx==1.5.6'])
# pip.main(['install', 'sphinx-gallery'])

# import sphinx_gallery
import subprocess

# import shlex

from subprocess import check_output, CalledProcessError

os.system('mkdir _static')
os.system('install -v -D css/custom.css _static/css/custom.css')
os.system('install -v -D js/expand_toc_tree.js _static/js/expand_toc_tree.js')

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('sphinx-apidoc'))
sys.path.insert(0, os.path.abspath('../doc'))
sys.path.insert(0, os.path.abspath('doc'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

print("Running sphinx-apidoc...")

cmd = "sphinx-apidoc --module-first -o " \
 + os.path.join(os.path.dirname(os.path.abspath(__file__)), '../odetoolbox') \
 + " " \
 + os.path.join(os.path.dirname(os.path.abspath(__file__)), '../odetoolbox')	# in-source generation of necessary .rst files

print("\tcmd = " + str(cmd))
os.system(cmd)


#
# 	copy static files/assets
#

import fnmatch
import os

static_docs_dir = os.path.dirname(os.path.abspath(__file__))
print("Searching in: " + str(static_docs_dir))
sys.path.insert(0, os.path.join(static_docs_dir, "odetoolbox"))
matches = []
for root, dirnames, filenames in os.walk(os.path.join(static_docs_dir, "..", "odetoolbox")):
    for filename in fnmatch.filter(filenames, '*.rst'):
            matches.append(os.path.join(root, filename))
#    for filename in fnmatch.filter(filenames, '*.pdf'):
#            matches.append(os.path.join(root, filename))
#    for filename in fnmatch.filter(filenames, '*.png'):
#            matches.append(os.path.join(root, filename))
print("Matches:")
print(matches)

#os.system("find ../odetoolbox")

for fn in matches:
	fn_from = fn
	fn_to = os.path.join(static_docs_dir, fn[len(static_docs_dir)+4:])
	print("From " + fn_from + " to " + fn_to)
	os.system('install -v -D ' + fn_from + " " + fn_to)
#os.system('for i in `find .. -name "*.rst"` ; do if [[ ${i} != *"sphinx-apidoc"* ]] ; then install -v -D ${i} ${i/\.\.\//}; fi ; done')


# The master toctree document.
master_doc = "index"

source_suffix = ['.rst']


# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax'
]

autodoc_default_flags = ['private-members', 'show-inheritance']

### show only __init__() and __str__() methods


mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# General information about the project.
project = u'ode-toolbox'
copyright = u'2004, nest-simulator'
author = u'nest-simulator'


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '2.1'
# The full version, including alpha/beta/rc tags
release = '2.1'
# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'manni'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# add numbered figure link
numfig = True

numfig_secnum_depth = (2)
numfig_format = {'figure': 'Figure %s', 'table': 'Table %s',
                 'code-block': 'Code Block %s'}
# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"
html_title = "ODE-toolbox"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'logo_only': True,
    'navigation_depth': 1,
    'collapse_navigation': False,
    'globaltoc_collapse': True,
    'globaltoc_maxdepth': 1
}


html_logo = "https://raw.githubusercontent.com/nest/ode-toolbox/master/doc/fig/ode-toolbox-logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['css', 'fig', 'js', '_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css'
]

html_js_files = [
    'js/expand_toc_tree.js'
]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'ode-toolbox-doc'

html_show_sphinx = False
html_show_copyright = False

# This way works for ReadTheDocs
# With this local 'make html' is broken!
github_doc_root = ''

intersphinx_mapping = {'https://docs.python.org/': None}


def skip(app, what, name, obj, would_skip, options):
    if name in ["__init__", "__str__"]:
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'ode-toolbox-doc.tex', u'ode-toolbox documentation',
     u'ode-toolbox documentation', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'ode-toolbox-doc', u'ode-toolbox documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ode-toolbox-doc', u'ode-toolbox documentation',
     author, 'ode-toolbox-doc', 'ode-toolbox documentation',
     'Miscellaneous'),
]

# -- Options for readthedocs ----------------------------------------------
# on_rtd = os.environ.get('READTHEDOCS') == 'True'
# if on_rtd:
#    html_theme = 'alabaster'
# else:
#    html_theme = 'nat'
