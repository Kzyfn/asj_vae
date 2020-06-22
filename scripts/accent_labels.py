import pandas as pd

import numpy as np


from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnmnkwii.io import hts
from os.path import join
from glob import glob
import pysptk
import pyworld
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists
import os
import sys


# In[7]:


def a1_a2_f2_extractor(str_label):
    index_mora = str_label.index("-")
    mora_str = str_label[index_mora + 1 : str_label.index("+")]

    if mora_str not in ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cli"]:
        return None

    index_a = str_label.index("/A:")
    index_a1 = str_label.index("+", index_a + 1)

    a1 = str_label[index_a + 3 : index_a1]

    a2 = str_label[index_a1 + 1 : str_label.index("+", index_a1 + 1)]

    index_f = str_label.index("/F")

    f_str = str_label[index_f : mora[2].index("/G")]
    f1 = f_str[f_str.index("/") + 1 : f_str.index("_")]
    f2 = f_str[f_str.index("_") + 1 : f_str.index("#")]

    return a1, a2, f2


# In[8]:


def accent_extractor(str_label):

    if a1_a2_f2_extractor(str_label) is None:
        return None

    a1, a2, f2 = a1_a2_f2_extractor(str_label)

    if a2 == "1":  # アクセント句の先頭
        if f2 == "1":  # 頭高型
            return 1
        else:  # 中高 or 平板
            return 0
    else:  # アクセント句の先頭以外
        if int(a1) <= 0:  #
            return 1

        else:
            return 0


import numpy as np
from nnmnkwii.io import hts

from tqdm import tqdm

paths = sorted(glob(join("../data/basic5000", "label_phone_align", "*.lab")))
os.makedirs("../data/basic5000/accemts")

for i, filepath in tqdm(enumerate(paths)):
    label = hts.load(filepath)

    accents = []

    for mora in label:
        str_label = mora[2]
        accent = accent_extractor(str_label)
        if accent is not None:
            accents.append(accent)

    np.savetxt(
        "..data/basic5000/accents/accents_" + "0" * (4 - len(str(i))) + str(i) + ".csv",
        accents,
    )

