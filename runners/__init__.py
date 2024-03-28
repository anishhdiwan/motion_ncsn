# Assuming that motion_ncsn is a submodule in the algo directory
import sys
import os
# MOTION_LIB_PATH = os.path.join(os.path.dirname(__file__),
#                                '../../../custom_envs')
# sys.path.append(MOTION_LIB_PATH)

# init.py is the first script to run when something is imported from this module.
# Adding the main module path so that relative imports from scripts outside this path work
NCSN_PATH = os.path.join(os.path.dirname(__file__),
                               '../')
sys.path.append(NCSN_PATH)

from runners.toy_runner import *
from runners.scorenet_runner import *
from runners.anneal_runner import *
from runners.baseline_runner import *
