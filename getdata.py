from pitchcharacteristic import *


def getdemo(i):
    # if(i<=4):
    wavpath = "/home/Coding/ConTunerData/demo/song/" + str(i) + ".wav"
    return wavpath
    # else:
    #     wavpath="/home/Coding/ConTunerData/demo/song/"+str(i-4)+"pro.wav"
    #     return wavpath


# amateur
def get_melpath(i, j):
    wav_a = "/home/Coding/ConTunerData/ama/jsp/" + str(i) + ".wav"
    wav_b = "/home/Coding/ConTunerData/ama/syf/" + str(i) + ".wav"
    wav_c = "/home/Coding/ConTunerData/ama/zjf/" + str(i) + ".wav"
    wav_d = "/home/Coding/ConTunerData/ama/zjh/" + str(i) + ".wav"
    if j == 1:
        return wav_a
    if j == 2:
        return wav_b
    if j == 3:
        return wav_c
    if j == 4:
        return wav_d
    pass


# amateur sp, f0
def get_path(i, j):
    wav_a = "/home/Coding/ConTunerData/ama/jsp/" + str(i) + ".wav"
    wav_b = "/home/Coding/ConTunerData/ama/syf/" + str(i) + ".wav"
    wav_c = "/home/Coding/ConTunerData/ama/zjf/" + str(i) + ".wav"
    wav_d = "/home/Coding/ConTunerData/ama/zjh/" + str(i) + ".wav"
    jspf0, jspsp, jspah = getsp(wav_a)
    syff0, syfsp, syfah = getsp(wav_b)
    zjff0, zjfsp, zjfah = getsp(wav_c)
    zjhf0, zjhsp, zjhah = getsp(wav_d)
    if j == 1:
        return jspf0, jspsp, jspah
    if j == 2:
        return syff0, syfsp, syfah
    if j == 3:
        return zjff0, zjfsp, zjfah
    if j == 4:
        return zjhf0, zjhsp, zjhah
    pass


# amateur MIDI
def get_MIDIpath(i, j):
    wav_a = "/home/Coding/ConTunerData/ama/a/midi/" + str(i) + ".csv"
    wav_b = "/home/Coding/ConTunerData/ama/b/midi/" + str(i) + ".csv"
    wav_c = "/home/Coding/ConTunerData/ama/c/midi/" + str(i) + ".csv"
    wav_d = "/home/Coding/ConTunerData/ama/d/midi/" + str(i) + ".csv"

    wav_a1 = "/home/Coding/ConTunerData/ama/a/" + str(i) + ".wav"
    wav_b1 = "/home/Coding/ConTunerData/ama/b/" + str(i) + ".wav"
    wav_c1 = "/home/Coding/ConTunerData/ama/c/" + str(i) + ".wav"
    wav_d1 = "/home/Coding/ConTunerData/ama/d/" + str(i) + ".wav"

    wav_a = getMIDI(wav_a, wav_a1)
    wav_b = getMIDI(wav_b, wav_b1)
    wav_c = getMIDI(wav_c, wav_c1)
    wav_d = getMIDI(wav_d, wav_d1)
    if j == 1:
        return wav_a
    if j == 2:
        return wav_b
    if j == 3:
        return wav_c
    if j == 4:
        return wav_d
    pass


# professional MIDI, envelope, pitch
def get_pro(i):
    prowav_path = "/home/Coding/ConTunerData/pro/" + str(i) + ".wav"
    promidi_path = "/home/Coding/ConTunerData/pro/" + str(i) + ".csv"
    prof0, prosp, proah = getsp(prowav_path)
    promidi = getMIDI(promidi_path, prowav_path)
    return prof0, prosp, promidi


# professional Mel
def get_melpropath(i):
    prowav_path = "/home/Coding/ConTunerData/pro/" + str(i) + ".wav"
    return prowav_path


def get_propredictor(i):
    prowav_path = "/home/Coding/ConTuner/pitchjj/" + str(i) + "pro.wav"
    promidi_path = "/home/Coding/ConTuner/pitchjj/" + str(i) + "pro.csv"
    prof0, prosp, proah = getsp(prowav_path)
    promidi = getMIDI(promidi_path, prowav_path)
    return prof0, prosp, promidi


def get_amapredictor(i):
    wav = "/home/Coding/ConTuner/pitchjj/" + str(i) + ".wav"

    return getsp(wav)


def get_midipredictor(i):
    wav_a = "/home/Coding/ConTuner/pitchjj/" + str(i) + ".csv"

    wav_a1 = "/home/Coding/ConTuner/pitchjj/" + str(i) + ".wav"

    wav_a = getMIDI(wav_a, wav_a1)

    return wav_a
