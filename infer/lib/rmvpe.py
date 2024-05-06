from io import BytesIO
import os
from typing import List, Optional, Tuple
import numpy as np
import torch

from infer.lib import jit

try:
    # Fix "Torch not compiled with CUDA enabled"
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import torch.nn as nn
import torch.nn.functional as F
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window

import logging