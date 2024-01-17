from net import DiffNet  #逆采样阶段，降噪器的模型结构
from getmel import *  #mel频谱的预处理
from diffusion import GaussianDiffusion #diffusion模型的中间过程，不包括训练和采样的执行


import argparse
import torch
import numpy
import logging
import sys
import os
from tqdm import tqdm



#设置一些超参数
parser = argparse.ArgumentParser("DiffBeautifier")
parser.add_argument('--save', type=str, default='/home/jishengpeng/NlpVoice/DiffBeautifer/result/cryresult', help='experiment name')
parser.add_argument('--hidden_size1', type=int, default=256, help="the size of hidden cell")
parser.add_argument('--audio_num_mel_bins', type=int, default=80)
parser.add_argument('--timesteps', type=int, default=100, help="the steps of the diffusion")
parser.add_argument('--timescale', type=int, default=1)
parser.add_argument('--diff_loss_type', type=str, default='l1')
parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--spec_max', type=str, default='[]')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--train_epoch', type=int, default=1000000)

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
    device=torch.device("cuda:0")
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

    #加载数据
    ji=r'/home/jishengpeng/NlpVoice/Data/data/cry.wav'


    mel , _= get_spectrograms(ji)
    mel=np.array(mel)
    

    #进行mel谱的绘制
    plt.figure(figsize=(16,8))
    librosa.display.TimeFormatter(lag=True)
    mel_img=librosa.display.specshow(mel, y_axis='mel', x_axis='s')#, fmax=8000
    plt.title(f'Mel-Spectrogram')
    plt.colorbar(mel_img,format='%+2.0f dB')
    plt.savefig("/home/jishengpeng/NlpVoice/DiffBeautifer/result/cryresult/image1.png")
    plt.close()

    mel=torch.from_numpy(mel)
    mel=mel.to(device)
    print("mel频谱的shape:",mel.shape)
    logging.info('mel频谱的shape:%s',mel.shape)

    #构造diffusion模型和去噪器网络
    denoise_model=DiffNet(80)
    denoise_model=denoise_model.to(device)

    diffusion_model = GaussianDiffusion(
        out_dims=args.audio_num_mel_bins,
        denoise_fn=denoise_model,   #传入wavenet
        timesteps=args.timesteps,
        time_scale=args.timescale,
        loss_type=args.diff_loss_type,  #l1
        spec_min=[], 
        spec_max=[],  #传进来[]，不知道是啥
    )
    diffusion_model=diffusion_model.to(device)

    ## 初始化损失函数
    loss_f = torch.nn.L1Loss().to(device) ## |X-Y| ## 可以换成 torch.nn.MSELoss()


    ## 神经网络的参数优化器
    optim  = torch.optim.Adam(denoise_model.parameters(),
                                          lr=1e-5,
                                          betas=(0.9, 0.999), eps=1e-09, weight_decay=1e-8)

    cond=torch.rand(size=(args.batch_size,mel.shape[0],mel.shape[1])).to(device)


    #开始训练
    for traini in range(args.epoch):
       
        input=mel.reshape(args.batch_size,1,mel.shape[0],mel.shape[1])  #[B,1,80,T]
        input=input.to(device)
        print("*****当前是第",traini,"次epoch*****")
        logging.info("*****当前是第%d次epoch*****",traini)
        #训练阶段，我在这边加了一个train_epoch
        for j in range(args.train_epoch):
            #构造模型的输入
            t = torch.randint(0, args.timesteps + 1, (args.batch_size,), device=device).long()  #这里可以保证每次时间t
            # print("t.shape,t",t.shape,t)
            # cond=torch.rand(size=(args.batch_size,mel.shape[0],mel.shape[1])).to(device)
            # print("con",cond.shape)
            diffusionz=mel.reshape(args.batch_size,mel.shape[1],mel.shape[0]) #[B,T_s,80]


            # print("diffusionz",diffusionz.shape)

            # Diffusion
            x_t = diffusion_model.diffuse_fn(diffusionz, t)      #[B,1,80,T]   # [B, T_s, 80]

            # print("x_t",x_t.shape)

            # Predict x_{start}
            x_0_pred = denoise_model(x_t, t.reshape(-1,1), cond)  #[B,1,80,T]
            # print("x_0_pred",x_0_pred.shape)

            loss_l1 = loss_f(x_0_pred,input)

            optim.zero_grad()  ## 梯度清空
            loss_l1.backward() ##  计算梯度
            optim.step()  ##  更新参数

            if(traini%1==0):
                loss_d = {}
                loss_d["epoch"] = traini
                loss_d["step"] = j
                loss_d["l1_loss"] = loss_l1.item()
                write_line2log(loss_d,"/home/jishengpeng/NlpVoice/DiffBeautifer/result/cryresult/train_log.txt",True)

        #推理阶段
        if(traini%10==0):
            t = args.timesteps  # reverse总步数
            shape = (cond.shape[0], 1, args.audio_num_mel_bins, cond.shape[2])
            # x = torch.randn(shape, device=device)  # noise
            x=x_t
            for i in tqdm(reversed(range(0, t)), desc='ProDiff sample time step', total=t):
                x = diffusion_model.p_sample(x, torch.full((args.batch_size,), i, device=device, dtype=torch.long), cond)  # x(mel), t, condition(phoneme)

            
            tmp=x.reshape(x.shape[2],x.shape[3])
            tmp=tmp.cpu().numpy()
            # print(tmp.shape)
        
            #进行mel谱的绘制
            plt.figure(figsize=(16,8))
            librosa.display.TimeFormatter(lag=True)
            mel_img=librosa.display.specshow(tmp, y_axis='mel', x_axis='s')#, fmax=8000
            plt.title(f'Mel-Spectrogram')
            plt.colorbar(mel_img,format='%+2.0f dB')
            plt.savefig("/home/jishengpeng/NlpVoice/DiffBeautifer/result/cryresult/imageepoch"+str(traini)+".png")
            plt.close()

            #合成一些语音听一听
            wav1 = melspectrogram2wav(tmp.T) # input size : (frames ,ndim)
            sr=16000
            outputfile="/home/jishengpeng/NlpVoice/DiffBeautifer/result/cryresult/stdioepoch"+str(traini)+".wav"
            soundfile.write(outputfile, wav1, sr)
            # print("finished change ")



    #Dataloader

    #开始模型的训练

    #开始模型的推理

    pass






if __name__=="__main__":
    main()