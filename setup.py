#!/usr/bin/env python
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.0'
DESCRIPTION = 'Unified Explanation Provider For CNNs'
LONG_DESCRIPTION = 'This package provides explanations for CNN predictions using LIME, Integrated Gradients, SHAP, and Anchors, combined with a unifying LISA method.'

setup(
    name="LISA_CNN_ExplainerV3",
    version=VERSION,
    author="Sudil H.P Abeyagunasekera",
    author_email="<sudilhasithaa51@gmail.com>",
    url="https://github.com/SudilHasitha/LISA_CNN_ExplainerV3",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'opencv-python',
        'numpy',
        'matplotlib',
        'shap',
        'alibi',
        'lime',
        'scikit-image'
    ],
    keywords=['LIME', 'Integrated gradients', 'SHAP', 'Anchors', 'Explainable AI', 'XAI', 'CNN Explainer', 'LISA'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)