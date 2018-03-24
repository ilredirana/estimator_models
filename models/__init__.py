import os
from inspect import getsourcefile

MODELS_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

print("Models dir is '%s'" % MODELS_DIR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(getsourcefile(lambda: 0))))

print("Base dir is '%s'" % BASE_DIR)
