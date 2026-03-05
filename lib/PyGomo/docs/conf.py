# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'PyGomo'
copyright = '2026, PyGomo Contributors'
author = 'PyGomo Contributors'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx_copybutton',       # Add copy button to code blocks
    'myst_parser',             # Parse Markdown
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Theme
html_theme = 'furo'  # Modern, clean theme
html_static_path = ['_static']
html_title = "PyGomo 0.1.0"
