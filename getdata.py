from pitchcharacteristic import *

def getdemo(i):
    # if(i<=4):
    wavpath="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/demo/song/"+str(i)+".wav"
    return wavpath
    # else:
    #     wavpath="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/demo/song/"+str(i-4)+"pro.wav"
    #     return wavpath


#得到业余的mel谱的路径
def get_melpath(i,j):
    jspwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/jsp/"+str(i)+".wav"
    syfwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/syf/"+str(i)+".wav"
    zjfwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjf/"+str(i)+".wav"
    zjhwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjh/"+str(i)+".wav"
    if(j==1):
        return jspwav
    if(j==2):
        return syfwav
    if(j==3):
        return zjfwav
    if(j==4):
        return zjhwav
    pass

#得到业余歌曲sp和f0的路径
def get_path(i,j):
    jspwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/jsp/"+str(i)+".wav"
    syfwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/syf/"+str(i)+".wav"
    zjfwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjf/"+str(i)+".wav"
    zjhwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjh/"+str(i)+".wav"
    jspf0,jspsp,jspah=getsp(jspwav)
    syff0,syfsp,syfah=getsp(syfwav)
    zjff0,zjfsp,zjfah=getsp(zjfwav)
    zjhf0,zjhsp,zjhah=getsp(zjhwav)
    if(j==1):
        return jspf0,jspsp,jspah
    if(j==2):
        return syff0,syfsp,syfah
    if(j==3):
        return zjff0,zjfsp,zjfah
    if(j==4):
        return zjhf0,zjhsp,zjhah
    pass

#得到业余MIDI的路径
def get_MIDIpath(i,j):
    jspwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/jsp/jspmidi/"+str(i)+".csv"
    syfwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/syf/syfmidi/"+str(i)+".csv"
    zjfwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjf/zjfmidi/"+str(i)+".csv"
    zjhwav="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjh/zjhmidi/"+str(i)+".csv"


    jspwav1="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/jsp/"+str(i)+".wav"
    syfwav1="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/syf/"+str(i)+".wav"
    zjfwav1="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjf/"+str(i)+".wav"
    zjhwav1="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/ama/zjh/"+str(i)+".wav"

    jspmidi=getMIDI(jspwav,jspwav1)
    syfmidi=getMIDI(syfwav,syfwav1)
    zjfmidi=getMIDI(zjfwav,zjfwav1)
    zjhmidi=getMIDI(zjhwav,zjhwav1)
    if(j==1):
        return jspmidi
    if(j==2):
        return syfmidi
    if(j==3):
        return zjfmidi
    if(j==4):
        return zjhmidi
    pass


#得到专业的MIDI和频谱包络和音高f0的路径
def get_pro(i):
    prowav_path="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/pro/"+str(i)+".wav"
    promidi_path="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/pro/"+str(i)+".csv"
    prof0,prosp,proah=getsp(prowav_path)
    promidi=getMIDI(promidi_path,prowav_path)
    return prof0,prosp,promidi

#得到专业的mel的路径
def get_melpropath(i):
    prowav_path="/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/pro/"+str(i)+".wav"
    return prowav_path




def get_propredictor(i):
    prowav_path="/home/jishengpeng/pitchjj/"+str(i)+"pro.wav"
    promidi_path="/home/jishengpeng/pitchjj/"+str(i)+"pro.csv"
    prof0,prosp,proah=getsp(prowav_path)
    promidi=getMIDI(promidi_path,prowav_path)
    return prof0,prosp,promidi

def get_amapredictor(i):
    wav="/home/jishengpeng/pitchjj/"+str(i)+".wav"
   
    zjhf0,zjhsp,zjhah=getsp(wav)

    return zjhf0,zjhsp,zjhah


def get_midipredictor(i):
    jspwav="/home/jishengpeng/pitchjj/"+str(i)+".csv"

    jspwav1="/home/jishengpeng/pitchjj/"+str(i)+".wav"


    jspmidi=getMIDI(jspwav,jspwav1)


    return jspmidi



