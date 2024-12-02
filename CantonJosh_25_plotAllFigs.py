# ==============================
# ======  Dependencies  ========
# ==============================
#
# PintoLab_dj 
# PintoLab_imagingAnalysis
# PintoLab_utils
# these can all be installed as packages


# ==============================
# ======  Import stuff  ========
# ==============================

from utils import connect_to_dj
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
VM = connect_to_dj.get_virtual_modules()


# ==============================
# ======  General info  ========
# ==============================
code_dir = "/Users/lpr6177/Documents/code/Canton-JoshEtAl2025/"
sys.path.insert(0,code_dir)


