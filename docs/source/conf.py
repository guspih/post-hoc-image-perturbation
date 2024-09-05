# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

project = 'Perturbation-based Post-hoc Explanations for Image Attribution'
copyright = '2024, Gustav Grund Pihlgren (guspih@github.com)'
author = 'Gustav Grund Pihlgren (guspih@github.com)'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver'
]
autosummary_generate=True
napoleon_custom_sections = [('Returns', 'params_style')]


templates_path = ['_templates']
exclude_patterns = ['docs']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
