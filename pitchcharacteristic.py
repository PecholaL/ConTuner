import pyworld as pw
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import torch
#这个用来提取MIDI，频谱包络，音高



out_path=r"/home/jishengpeng/NlpVoice/Diffusionmel/result/tmp/001.wav"


def getsp(wav_path):

    # x, fs = sf.read(wav_path)
    x, fs = librosa.load(wav_path)
    # print("*****",fs)
    x = x.astype(np.double)


    # _f0_h, t_h = pw.harvest(x, fs)   #每隔0.005秒进行一下提取
    _f0_h, t_h = pw.dio(x, fs)
    # print(_f0_h.shape,t_h.shape,_f0_h,t_h)

    #得到基频
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)

    # print(f0_h.shape)

    sp_h = pw.cheaptrick(x, f0_h, t_h, fs)  #频谱包络是一个二维的矩阵

    # print(sp_h.shape)

    ap_h = pw.d4c(x, f0_h, t_h, fs)

    # #合成的一个函数
    # y_h = pw.synthesize(f0_h, sp_h, ap_h, fs, pw.default_frame_period)
    # # librosa.output.write_wav('result/y_harvest_with_f0_refinement.wav', y_h, fs)

    # sf.write(out_path, y_h, fs)

    # print("******")

    #将其变成float
    f0_h = f0_h.astype(np.float)
    sp_h = sp_h.astype(np.float)


    # print(f0_h.dtype)



    # #将其保留小数点两位,太耗时了
    # height,weight=sp_h.shape

    # for i in range(height):
    #     print(i)
    #     f0_h[i]=round(f0_h[i],2)
    #     for j in range(weight):
    #         sp_h[i][j]=round(sp_h[i][j],2)



    return f0_h , sp_h ,ap_h   #第一个维度是一个一维的向量, 这是一个m*n的维度,其中m这个值是和一维的f0的值是一样的


def getf0(wav_path):
    # x, fs = sf.read(wav_path)
    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)


    # _f0_h, t_h = pw.harvest(x, fs)   #每隔0.005秒进行一下提取
    _f0_h, t_h = pw.dio(x, fs)
    # print(_f0_h.shape,t_h.shape,_f0_h,t_h)

    #得到基频
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)

    return f0_h



def getMIDI(MIDI_path,wav_path):
    #这里将MIDI数据同样处理成跟音频一样的【M，】维一维向量
    df=pd.read_csv(MIDI_path)
    content=df.values
    content=np.array(content)
    height,weight=content.shape

    # print(content)



    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)
    _f0_h, t_h = pw.dio(x, fs)

    maxlen=t_h.shape[0]

    # print(maxlen)

    res=[]  #返回的最终结果

    for i in range(maxlen):
        j=0
        while(j<height):
            if(t_h[i]>content[j][0]+content[j][2]):
                j=j+1
                continue

            if(t_h[i]<content[j][0]):
                res.append(0)
                break
            else:
                res.append(round(content[j][1],0))
                break

        if(j==height):
            res.append(0)

        pass

    res=np.array(res)

    # print(res.shape)







    # height,weight=content.shape
    # #保留两位小数,只需要那个频率值和持续时间
    # tmp=[]
    # for i in range(height):
    #     tmptmp=[]
    #     # for j in range(weight):
    #     tmptmp.append(round(content[i][1],2))
    #     tmptmp.append(round(content[i][2],2))
    #     tmp.append(tmptmp)
    # content=np.array(tmp)
    res=torch.Tensor(res)
    # print(content.shape,content)
    # print(content,content.shape)
    # content=content.double()
    # print(content,content.shape)


    return res  #这是一个n*1的维度
    pass



def main():

    wav_path=r"/home/jishengpeng/NlpVoice/Data/data/cry.wav"
    # for i in range(8,9):
    #     wav_path="/home/jishengpeng/hijsp/"+str(i)+"ama.wav"
    # MIDI_path=r"/home/jishengpeng/NlpVoice/Data/DiffBeautifierData/pro/1.csv" 
    # f0=getf0(wav_path)
    # print(f0,f0.shape)
    f0,sp,_=getsp(wav_path)

    print(f0.shape[0])

    # print(f0,f0.shape)

    # maxlen=1600

    # addlen=maxlen-f0.shape[0]

    # f0=f0.tolist()

    # for addi in range(addlen):
    #     f0.append(0)
    
    # f0=np.array(f0)
    

    # x=f0.reshape(1,80,-1)
    # # # print(sp,sp.shape)
    # print(f0,f0.shape)
    # print(x,x.shape)
    # getMIDI(MIDI_path,wav_path) 
    # pass



if __name__=="__main__":
    main()
