from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Unified Explanation Provider For CNNs'
LONG_DESCRIPTION = 'This package allows to get explantions for the predictions made by the CNNs using existing LIME, Integrated Gradients,SHAP and Anchors with newly introduced unifiying method in this package.'

# Setting up
setup(
    name="ExplainLISA",
    version=VERSION,
    author="Sudil H.P Abeyagunasekera",
    author_email="<sudilhasithaa51@gmail.com>",
    url="https://github.com/SudilHasitha/ExplainLISA",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['opencv-python', 'tensorflow', 'numpy','pandas','matplotlib','alibi[ray]','lime','shap','scikit-image'],
    keywords=['LIME', 'Integrated gradients', 'SHAP', 'Anchors', 'Explainable AI', 'XAI','CNN Explainer'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)