1. Tensorflow 2.1.0 has no candidate because of bazel building with python2.7
-  Python3 does not fix the problem. pip3_import cannot find tensorflow>2.0
-  Workaround, remove tensorflow from requirements, install manually

2. Imported clgen.runfiles/phd/deeplearning/clgen are incompatible with tensorflow 2.1.0
-  Import manually instead of http pulling and convert compatibility
