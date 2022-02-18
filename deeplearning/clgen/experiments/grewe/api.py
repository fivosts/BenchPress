"""
API to communicate with legacy 'preamble.py' and 'model.py'
of Grewe's et al. predictive model (CGO 2013).

This API is used to convert modernized database groups
to the expected csv files by the script and also fill in
missing cldrive data.
"""
import pathlib

from deeplearning.clgen.experiments import public

"""
1. You may insert database groups as usual to convert to csv
2. You need to introduce a systematic way to insert the amd/nvidia/clgen csv's from clgen's artifacts.
  a) Could be protobuf path arguments pointing to results workspace
"""

