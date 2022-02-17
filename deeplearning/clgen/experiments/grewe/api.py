"""
API to communicate with legacy 'preamble.py' and 'model.py'
of Grewe's et al. predictive model (CGO 2013).

This API is used to convert modernized database groups
to the expected csv files by the script and also fill in
missing cldrive data.
"""
import pathlib

from deeplearning.clgen.experiments import public