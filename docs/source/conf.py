# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add the project root and docs parent to path
docs_dir = Path(__file__).parent.parent
project_root = docs_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(docs_dir))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import importlib.metadata

# Get metadata from package
metadata = importlib.metadata.metadata('piedomains')
project = metadata['Name']
release = metadata['Version']

# Extract authors from metadata
authors_list = []
for author in metadata.get_all('Author') or []:
    authors_list.append(author)
for author_email in metadata.get_all('Author-Email') or []:
    # Parse "Name <email>" format
    if '<' in author_email and '>' in author_email:
        name = author_email.split('<')[0].strip()
        if name:
            authors_list.append(name)

author = ', '.join(set(authors_list)) if authors_list else 'piedomains developers'

copyright = f'{datetime.datetime.now().year}, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
]

# Mock imports for modules that aren't needed for docs
autodoc_mock_imports = [
    'tensorflow',
    'keras',
    'playwright',
    'playwright.sync_api',
    'playwright.async_api',
    'selenium',
    'webdriver_manager',
    'nltk',
    'scikit-learn',
    'sklearn',
    'joblib',
    'litellm'
]

# Source file configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = project
html_static_path = ['_static']

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#336790",
        "color-brand-content": "#336790",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4db8ff",
        "color-brand-content": "#4db8ff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
}

# Custom sidebar templates for furo theme
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
