from net import *  #逆采样阶段，降噪器的模型结构
from getmel import *  #mel频谱的预处理
from diffusion import GaussianDiffusion #diffusion模型的中间过程，不包括训练和采样的执行
from getdata import *  #加载f0,sp,midi

import argparse
import torch
import numpy
import logging
import sys
import os
from tqdm import tqdm



#设置一些超参数
parser = argparse.ArgumentParser("DiffBeautifier")
parser.add_argument('--save', type=str, default='/home/jishengpeng/NlpVoice/DiffBeautifer/result/n1result', help='experiment name')
parser.add_argument('--hidden_size1', type=int, default=256, help="the size of hidden cell")
parser.add_argument('--audio_num_mel_bins', type=int, default=80)
parser.add_argument('--timesteps', type=int, default=10, help="the steps of the diffusion")
parser.add_argument('--timescale', type=int, default=1)
parser.add_argument('--diff_loss_type', type=str, default='l1')
parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--spec_max', type=str, default='[]')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--train_epoch', type=int, default=10000)
parser.add_argument('--maxlen',type=int, default=1600)
parser.add_argument('--datai',type=int,default=300)
parser.add_argument('--dataj',type=int,default=4)
parser.add_argument('--fs',type=int,default=22050)
parser.add_argument('--pitch_vec_maxlen',type=int,default=60000)

args = parser.parse_args()


#日志
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


#定义设备
if(torch.cuda.is_available()):
    device=torch.device("cuda:4")
else:
    device=torch.device("cpu")


#保存训练过程
def write_line2log(log_dict: dict, filedir, isprint: True):
    strp = ''
    with open(filedir, 'a', encoding='utf-8') as f:
        for key, value in log_dict.items():
            witem = '{}'.format(key) + ':{},'.format(value)
            strp += witem
        f.write(strp)
        f.write('\n')
    if isprint:
        print(strp)
    pass


def main():

    # #进行mel谱的绘制
    # plt.figure(figsize=(16,8))
    # librosa.display.TimeFormatter(lag=True)
    # mel_img=librosa.display.specshow(mel, y_axis='mel', x_axis='s')#, fmax=8000
    # plt.title(f'Mel-Spectrogram')
    # plt.colorbar(mel_img,format='%+2.0f dB')
    # plt.savefig("/home/jishengpeng/NlpVoice/DiffBeautifer/result/tmpresult/image1.png")
    # plt.close()


    #构造diffusion模型和去噪器模型
    denoise_model=DiffNetCon(80)
    denoise_model=denoise_model.to(device)

    diffusion_model = GaussianDiffusion(
        out_dims=args.audio_num_mel_bins,
        denoise_fn=denoise_model,   #传入wavenet
        timesteps=args.timesteps,
        time_scale=args.timescale,
        loss_type=args.diff_loss_type,  #l1
        spec_min=[], 
        spec_max=[], 
    )
    diffusion_model=diffusion_model.to(device)

    #音高预测器的模型
    pitch_model=PitchNet()
    pitch_model=pitch_model.to(device)

    ## 初始化损失函数
    loss_f = torch.nn.L1Loss().to(device) ## |X-Y| ## 可以换成 torch.nn.MSELoss()


    ## 神经网络的参数优化器
    optim  = torch.optim.Adam(denoise_model.parameters(),
                                          lr=1e-5,
                                          betas=(0.9, 0.99), eps=1e-09, weight_decay=1e-8)

    # cond=torch.rand(size=(args.batch_size,mel.shape[0],20)).to(device)  #[B,80,T]

    


    #开始训练
    for traini in range(args.epoch):
       
        
        # print("*****当前是第",traini,"次epoch*****")
        logging.info("*****当前是第%d次epoch*****",traini)
        #训练阶段，我在这边加了一个train_epoch，因为要随机t
        for j in range(args.train_epoch):
            sumloss=0

            #得到con
            for listi in range(args.datai):
#                 prof0,prosp,promidi=get_pro(listi+1)

                for listj in range(args.dataj):

                    logging.info("*****训练阶段,当前是第%d个step,第%d首歌,业余歌唱家是%d*****",j,listi+1,listj+1)
                    amaf0,amasp,amaah=get_path(listi+1,listj+1)
                    amamidi=get_MIDIpath(listi+1,listj+1)

                    #业余的歌手的歌曲
                    amasp_trans=amasp.reshape(1,amasp.shape[0],amasp.shape[1])
                    amamidi_trans=amamidi.reshape(1,amamidi.shape[0],1)
                    amaf0_trans=amaf0.reshape(1,amaf0.shape[0])


                    #将其放到gpu上

                    amaf0_trans=torch.Tensor(amaf0_trans)
                    amasp_trans=torch.Tensor(amasp_trans)
                    # amamidi_trans=torch.from_numpy(amamidi_trans)

                    amaf0_trans=amaf0_trans.float().to(device)
                    amasp_trans=amasp_trans.float().to(device)
                    amamidi_trans=amamidi_trans.float().to(device)

                    # print(amaf0_trans.dtype,amasp_trans.dtype,amamidi_trans.dtype)                    

                    pref0=pitch_model(amasp_trans,amamidi_trans)
                    loss_pitch=loss_f(pref0,amaf0_trans)
                    # sumloss=sumloss+loss_pitch

                    #进行转化
                    addlen=args.pitch_vec_maxlen-amaf0.shape[0]
                    f0=pref0.cpu()
                    f0=f0[0,:]
                    f0=f0.tolist()
                    #时间上
                    for addi in range(addlen):
                        f0.append(0)                   
                    f0=np.array(f0)
                    cond=f0.reshape(1,80,-1)  #[1,80,750]
                    cond=torch.Tensor(cond).float().to(device)

                    #构造模型t的输入和mel
                    t = torch.randint(0, args.timesteps + 1, (args.batch_size,), device=device).long()  #这里可以保证每次时间t
                    print("t.shape,t",t.shape,t)
                    # cond=torch.rand(size=(args.batch_size,mel.shape[0],mel.shape[1])).to(device)
                    print("con",cond.shape)

                    ji=get_melpath(listi+1,listj+1)
                    mel , _= get_spectrograms(ji)
                    mel=torch.from_numpy(mel)
                    mel=mel.to(device)
                    # print("mel频谱的shape:",mel.shape)
                    # logging.info('mel频谱的shape:%s',mel.shape)
                    input=mel.reshape(args.batch_size,1,mel.shape[0],mel.shape[1])  #[B,1,80,T]
                    input=input.to(device)
                    diffusionz=mel.reshape(args.batch_size,mel.shape[1],mel.shape[0]) #[B,T_s,80]


                    # print("diffusionz",diffusionz.shape)

                    # Diffusion
                    x_t = diffusion_model.diffuse_fn(diffusionz, t)      #[B,1,80,T]   # [B, T_s, 80]

                    print("x_t",x_t.shape)

                    # Predict x_{start}
                    x_0_pred = denoise_model(x_t, t.reshape(-1,1), cond)  #[B,1,80,T]
                    print("x_0_pred",x_0_pred.shape)

                    loss_l1 = loss_f(x_0_pred,input)
                    loss_final=loss_l1+0.0005*loss_pitch
                    sumloss=sumloss+loss_final

                    optim.zero_grad()  ## 梯度清空
                    loss_final.backward() ##  计算梯度
                    optim.step()  ##  更新参数

                    if(traini%5==0):
                        loss_d = {}
                        loss_d["epoch"] = traini
                        loss_d["step"] = j
                        loss_d["l1_loss"] = loss_final.item()
                        loss_d["mel"]=loss_l1.item()
                        loss_d["pitch"]=loss_pitch.item()
                        logging.info("*****业余歌曲损失%s*****",loss_d)

#                 #专业的歌曲的音高也可以进行训练
#                 #专业的歌手的歌曲
#                 prosp_trans=prosp.reshape(1,prosp.shape[0],prosp.shape[1])
#                 promidi_trans=promidi.reshape(1,promidi.shape[0],1)
#                 prof0_trans=prof0.reshape(1,prof0.shape[0])

#                 #将其放到gpu上

#                 prof0_trans=torch.Tensor(prof0_trans)
#                 prosp_trans=torch.Tensor(prosp_trans)
#                 # promidi_trans=torch.from_numpy(promidi_trans)

#                 prof0_trans=prof0_trans.float().to(device)
#                 prosp_trans=prosp_trans.float().to(device)
#                 promidi_trans=promidi_trans.float().to(device)

#                 pref0=pitch_model(prosp_trans,promidi_trans)
#                 loss_pitch=loss_f(pref0,prof0_trans)

#                 #进行转化
#                 addlen=args.pitch_vec_maxlen-prof0.shape[0]
#                 f0=pref0.cpu()
#                 f0=f0[0,:]
#                 f0=f0.tolist()
#                 for addi in range(addlen):
#                     f0.append(0)                   
#                 f0=np.array(f0)
#                 cond=f0.reshape(1,80,-1)
#                 cond=torch.Tensor(cond).float().to(device)

#                 #构造模型t的输入
#                 t = torch.randint(0, args.timesteps + 1, (args.batch_size,), device=device).long()  #这里可以保证每次时间t
#                 # print("t.shape,t",t.shape,t)
#                 # cond=torch.rand(size=(args.batch_size,mel.shape[0],mel.shape[1])).to(device)
#                 # print("con",cond.shape)


#                 ji=get_melpropath(listi+1)
#                 mel , _= get_spectrograms(ji)
#                 mel=torch.from_numpy(mel)
#                 mel=mel.to(device)
#                 # print("mel频谱的shape:",mel.shape)
#                 # logging.info('mel频谱的shape:%s',mel.shape)
#                 input=mel.reshape(args.batch_size,1,mel.shape[0],mel.shape[1])  #[B,1,80,T]
#                 input=input.to(device)
#                 diffusionz=mel.reshape(args.batch_size,mel.shape[1],mel.shape[0]) #[B,T_s,80]

#                 # print("diffusionz",diffusionz.shape)
#                 # Diffusion
#                 x_t = diffusion_model.diffuse_fn(diffusionz, t)      #[B,1,80,T]   # [B, T_s, 80]

#                 # print("x_t",x_t.shape)

#                 # Predict x_{start}
#                 x_0_pred = denoise_model(x_t, t.reshape(-1,1), cond)  #[B,1,80,T]
#                 # print("x_0_pred",x_0_pred.shape)

#                 loss_l1 = loss_f(x_0_pred,input)
#                 loss_final=loss_l1+0.0005*loss_pitch
#                 sumloss=sumloss+loss_final

#                 optim.zero_grad()  ## 梯度清空
#                 loss_final.backward() ##  计算梯度
#                 optim.step()  ##  更新参数

#                 if(traini%5==0):
#                     loss_d = {}
#                     loss_d["epoch"] = traini
#                     loss_d["step"] = j
#                     loss_d["l1_loss"] = loss_final.item()
#                     loss_d["mel"]=loss_l1.item()
#                     loss_d["pitch"]=loss_pitch.item()
#                     logging.info("*****专业歌曲损失%s*****",loss_d)

#             logging.info("*****第%d次epoch第%d次step的损失为%f*****",traini,j,sumloss)


        #推理阶段
        with torch.no_grad():
            if(traini%10==0):

                for listi in range(args.datai,400):
                    prof0,prosp,promidi=get_pro(listi+1)

                    for listj in range(args.dataj):
                        logging.info("*****推理阶段,当前是第%d首歌,业余歌唱家是%d*****",listi+1,listj+1)
                        amaf0,amasp,amaah=get_path(listi+1,listj+1)

                        #业余的歌手的歌曲
                        amasp_trans=amasp.reshape(1,amasp.shape[0],amasp.shape[1])
                        promidi_trans=promidi.reshape(1,promidi.shape[0],1)

                        #将其放到gpu上
                        amasp_trans=torch.Tensor(amasp_trans)

                        amasp_trans=amasp_trans.float().to(device)
                        promidi_trans=promidi_trans.float().to(device)

                        pref0=pitch_model(amasp_trans,promidi_trans)

                        # print("***pref0.shape",pref0.shape)

                        #进行转化
                        addlen=args.pitch_vec_maxlen-pref0.shape[1]
                        f0=pref0[0,:]

                        # for addi in range(pref0.shape[1]):
                        #     f0.append(pref0[0][addi])

                        # print("***f0.shape",f0.shape)

                        f0=f0.tolist()
                        for addi in range(addlen):
                            f0.append(0)                   
                        f0=np.array(f0)
                        cond=f0.reshape(1,80,-1)
                        cond=torch.Tensor(cond).float().to(device)


                        ji=get_melpath(listi+1,listj+1)
                        mel , _= get_spectrograms(ji)
                        mel=np.array(mel)


                        #进行mel谱的绘制
                        plt.figure(figsize=(16,8))
                        librosa.display.TimeFormatter(lag=True)
                        mel_img=librosa.display.specshow(mel, y_axis='mel', x_axis='s')#, fmax=8000
                        plt.title(f'Mel-Spectrogram')
                        plt.colorbar(mel_img,format='%+2.0f dB')
                        plt.savefig(args.save+"/mel/"+str(listi+1)+"_"+str(listj+1)+"image1.png")
                        plt.close()


                        mel=torch.from_numpy(mel)
                        mel=mel.to(device)
                        # print("mel频谱的shape:",mel.shape)
                        # logging.info('mel频谱的shape:%s',mel.shape)
                        input=mel.reshape(args.batch_size,1,mel.shape[0],mel.shape[1])  #[B,1,80,T]
                        input=input.to(device)
                        diffusionz=mel.reshape(args.batch_size,mel.shape[1],mel.shape[0]) #[B,T_s,80]

                        t = torch.randint(args.timesteps, args.timesteps + 1, (args.batch_size,), device=device).long()  #这里可以保证每次时间t
                        # # print("***",t,t[0])
                        # x = torch.randn(shape, device=device)  # noise
                        x_t = diffusion_model.diffuse_fn(diffusionz, t)
                        x=x_t.to(device)
                        t = args.timesteps  # reverse总步数
                        for i in tqdm(reversed(range(0, t)), desc='ProDiff sample time step', total=t):
                            x = diffusion_model.p_sample(x, torch.full((args.batch_size,), i, device=device, dtype=torch.long), cond)  # x(mel), t, condition(phoneme)

                
                        tmp=x.reshape(x.shape[2],x.shape[3])
                        tmp=tmp.cpu().numpy()
                        print(tmp.shape)
                    
                        #进行mel谱的绘制
                        plt.figure(figsize=(16,8))
                        librosa.display.TimeFormatter(lag=True)
                        mel_img=librosa.display.specshow(tmp, y_axis='mel', x_axis='s')#, fmax=8000
                        plt.title(f'Mel-Spectrogram')
                        plt.colorbar(mel_img,format='%+2.0f dB')
                        plt.savefig(args.save+"/mel/"+str(listi+1)+"_"+str(listj+1)+"imageepoch"+str(traini)+".png")
                        plt.close()

                        #合成一些语音听一听
                        wav1 = melspectrogram2wav(tmp.T) # input size : (frames ,ndim)
                        outputfile=args.save+"/"+str(listi+1)+"_"+str(listj+1)+"stdioepoch"+str(traini)+".wav"
                        soundfile.write(outputfile, wav1, args.fs)
                        # print("finished change ")


    pass

if __name__=="__main__":
    main()
