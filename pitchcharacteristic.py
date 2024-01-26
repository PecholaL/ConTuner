""" extraction of MIDI, spectral envelope and pitch label
"""

import pyworld as pw
import librosa
import numpy as np
import pandas as pd
import torch


test_out_path = r"/home/Coding/ConTuner/NlpVoice/Diffusionmel/result/tmp/001.wav"


def getsp(wav_path):
    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)

    # extract fundamental frequency
    _f0_h, t_h = pw.dio(x, fs)

    # modify f0
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)

    # extract spectral envelope
    sp_h = pw.cheaptrick(x, f0_h, t_h, fs)

    # extract aperiodic parameter
    ap_h = pw.d4c(x, f0_h, t_h, fs)

    """ 
    # synthesize original audio with the extracted components
    y_h = pw.synthesize(f0_h, sp_h, ap_h, fs, pw.default_frame_period)
    librosa.output.write_wav('result/y_harvest_with_f0_refinement.wav', y_h, fs)
    sf.write(test_out_path, y_h, fs)
    """

    f0_h = f0_h.astype(np.float)
    sp_h = sp_h.astype(np.float)

    return f0_h, sp_h, ap_h


def getf0(wav_path):
    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)

    _f0_h, t_h = pw.dio(x, fs)
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)

    return f0_h


def getMIDI(MIDI_path, wav_path):
    # shape MIDI to 1-D data (m,)
    df = pd.read_csv(MIDI_path)
    content = df.values
    content = np.array(content)
    height, _ = content.shape

    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)
    _, t_h = pw.dio(x, fs)

    maxlen = t_h.shape[0]

    res = []

    for i in range(maxlen):
        j = 0
        while j < height:
            if t_h[i] > content[j][0] + content[j][2]:
                j = j + 1
                continue

            if t_h[i] < content[j][0]:
                res.append(0)
                break
            else:
                res.append(round(content[j][1], 0))
                break

        if j == height:
            res.append(0)
        pass

    res = np.array(res)
    res = torch.Tensor(res)

    return res
