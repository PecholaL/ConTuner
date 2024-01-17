import librosa
import librosa.display
import soundfile
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal
import copy

sr = 16000 # Sample rate.
n_fft = 800 # fft points (samples)
hop_length = 200 # samples.
win_length = 800 # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20
top_db = 15

def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=16000)
    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # # Transpose
    # mel = mel.T.astype(np.float32)  # (T, n_mels)
    # mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag # (80,T)

def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram'''
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db

    # to amplitude  /-----/  the reverse of # to decibel
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(sr, n_fft, n_mels)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # c
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)



def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

if __name__ == '__main__':
    ## 给定一条语音
    # p = r'D:\语音数据\num_train\000909.wav'
    ji=r'/home/jishengpeng/月亮之上2.wav'
    outputfile=r'/home/jishengpeng/NlpVoice/DiffBeautifer/result/cryresult/crytest.wav'
    aa ,_= get_spectrograms(ji)



    # #画出原始语音
    # # Loading sound file
    # y, sr = librosa.load(p, sr=16000)
    # # Trimming
    # y, _ = librosa.effects.trim(y, top_db=top_db)
    # plt.figure()
    # plt.title("origin wavform")
    # plt.plot(y)
    # plt.show()


    # aa = get_spectrograms(p)
    # print(np.array(aa).shape)
    # print("melspec size: {} , stft spec size:{}".format(aa[0].shape,aa[1].shape))
    # # size : (frames, ndim)
    # # plt.figure()
    # # plt.title("oringin melspec")
    # # plt.imshow(aa[0],cmap='Greens')
    # # plt.show()
    # # plt.savefig("/home/jishengpeng/NlpVoice/Diffusionmel/result/test1.png")
    # # plt.close()


    # #进行mel谱的绘制
    # plt.figure(figsize=(16,8))
    # # kernel = np.ones((3,3),np.uint8)  ,
    # # mel_spect = cv2.morphologyEx(mel_spect, cv2.MORPH_OPEN, kernel)


    print(aa.shape)

    # librosa.display.TimeFormatter(lag=True)
    # mel_img=librosa.display.specshow(aa[0], y_axis='mel', x_axis='s')#, fmax=8000

    # plt.title(f'Mel-Spectrogram-jsptestforward')
    # plt.colorbar(mel_img,format='%+2.0f dB')
    # plt.savefig("/home/jishengpeng/NlpVoice/Diffusionmel/result/short/image1.png")
    # plt.close()

    # # plt.figure()
    # # plt.title("oringin stft mag")
    # # plt.imshow(aa[1],cmap='Greens')
    # # plt.show()
    # # plt.savefig("/home/jishengpeng/NlpVoice/Diffusionmel/result/test2.png")
    # # plt.close()

    ## 将 melspec  合成为语音。并和原始语音做比较
    print(aa[0].shape)
    wav1 = melspectrogram2wav(aa.T) # input size : (frames ,ndim)
    # print(wav1)
    # plt.figure()
    # plt.title("mel2wav: wavform")
    # plt.plot(wav1)
    # plt.show()

    # soundfile.write(p.replace('.w','_gff.w'), wav1, sr)
    soundfile.write(outputfile, wav1, 16000)
    print("finished change ")


    # ###  画出 转化语音的谱
    # aa = get_spectrograms(p.replace('.w','_gff.w'))
    # plt.figure()
    # plt.title("mel2wav melspec")
    # plt.imshow(aa[0], cmap='Greens')
    # plt.show()

    # plt.figure()
    # plt.title("mel2wav stft mag")
    # plt.imshow(aa[1], cmap='Greens')
    # plt.show()




