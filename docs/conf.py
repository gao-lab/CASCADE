# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import cascade

project = cascade.name
version = cascade.__version__
release = cascade.__version__
author = "Zhi-Jie Cao"
copyright = "Gao Lab, 2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.programoutput",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_show_sourcelink = True
set_type_checking_flag = True
napoleon_use_rtype = False
typehints_use_rtype = True
typehints_fully_qualified = False
always_use_bars_union = True
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = False
autodoc_default_options = {"autosummary": True}
nbsphinx_execute = "never"


intersphinx_mapping = dict(
    python=("https://docs.python.org/3/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    plotly=("https://plotly.com/python-api-reference/", None),
    networkx=("https://networkx.org/documentation/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    torch=("https://pytorch.org/docs/stable/", None),
    pytorch_lightning=("https://lightning.ai/docs/pytorch/stable/", None),
)


def skip_internal(app, what, name, obj, skip, options):
    if hasattr(obj, "_internal"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_internal)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
