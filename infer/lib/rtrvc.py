from io import BytesIO
import os
import sys
import traceback
from infer.lib import jit
from infer.lib.jit.get_synthesizer import get_synthesizer
from time import time as ttime
import fairseq
import faiss
import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcrepe
from torchaudio.transforms import Resample