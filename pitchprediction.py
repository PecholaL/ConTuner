#predict rhe pitch
import torch
import argparse
import logging
import sys
import os
import pyworld as pw
import numpy as np

from net import PitchNet
from getdata import *


#设置一些超参数
parser = argparse.ArgumentParser("PitchPrediction")
parser.add_argument('--save', type=str, default='/home/jishengpeng/NlpVoice/DiffBeautifer/result/pitchresult', help='experiment name')
#parser.add_argument('--output_ama', type=str, default='/home/jishengpeng/NlpVoice/DiffBeautifer/result/pitchresult')
parser.add_argument('--epoch',type=int,default=200)
# parser.add_argument('--trainepoch',type=int,default=10,help='moretrain')
parser.add_argument('--datai',type=int,default=2)
parser.add_argument('--dataj',type=int,default=4)
parser.add_argument('--fs',type=int,default=22050)

args = parser.parse_args()

#定义设备
if(torch.cuda.is_available()):
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")

#日志，设置模式为追加
log_format = '时间:%(asctime)s  日志信息:%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p',filemode='w')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():

    #音高预测器的模型
    pitch_model=PitchNet()
    pitch_model=pitch_model.to(device)


    ## 初始化损失函数
    loss_f = torch.nn.L1Loss().to(device) ## |X-Y| ## 可以换成 torch.nn.MSELoss()


    ## 神经网络的参数优化器
    optim  = torch.optim.Adam(pitch_model.parameters(),
                                          lr=5e-4,
                                          betas=(0.9, 0.999), eps=1e-09, weight_decay=1e-8)


    #模型训练，注意模型的输入要是float32,即Torch.Tensor(),最后还需要用.float()来控制
    for epochi in range(args.epoch):
        logging.info("*****当前是第%d次epoch*****",epochi)
        sumloss=0  #记录总的损失值
        for listi in range(args.datai):
            prof0,prosp,promidi=get_pro(listi+1)

            for listj in range(args.dataj):
                logging.info("*****训练阶段,当前是第%d首歌,业余歌唱家是%d*****",listi+1,listj+1)
                amaf0,amasp,amaah=get_path(listi+1,listj+1)
                amamidi=get_MIDIpath(listi+1,listj+1)
                

                #先生成业余自己提取特征自己合成的歌曲
                if(epochi==1):
                    out_path=args.save+"/"+str(listi+1)+"num_"+str(listj+1)+".wav"
                    y_h = pw.synthesize(amaf0, amasp, amaah, args.fs, pw.default_frame_period)
                    # librosa.output.write_wav('result/y_harvest_with_f0_refinement.wav', y_h, fs)
                    sf.write(out_path, y_h, args.fs)
                
                #进行模型训练


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
                loss_l1=loss_f(pref0,amaf0_trans)
                sumloss=sumloss+loss_l1

                optim.zero_grad()  ## 梯度清空
                loss_l1.backward() ##  计算梯度
                optim.step()  ##  更新参数

                #记录损失值
                if(epochi%10==0):
                    loss_d = {}
                    loss_d["epoch"] = epochi
                    loss_d["songnum"] = listi+1
                    loss_d["peoplenum"]=listj+1
                    loss_d["l1_loss"] = loss_l1.item()
                    logging.info("*****业余歌曲损失%s*****",loss_d)

            #专业的歌手的歌曲
            prosp_trans=prosp.reshape(1,prosp.shape[0],prosp.shape[1])
            promidi_trans=promidi.reshape(1,promidi.shape[0],1)
            prof0_trans=prof0.reshape(1,prof0.shape[0])

            #将其放到gpu上

            prof0_trans=torch.Tensor(prof0_trans)
            prosp_trans=torch.Tensor(prosp_trans)
            # promidi_trans=torch.from_numpy(promidi_trans)

            prof0_trans=prof0_trans.float().to(device)
            prosp_trans=prosp_trans.float().to(device)
            promidi_trans=promidi_trans.float().to(device)

            pref0=pitch_model(prosp_trans,promidi_trans)
            loss_l1=loss_f(pref0,prof0_trans)
            sumloss=sumloss+loss_l1

            optim.zero_grad()  ## 梯度清空
            loss_l1.backward() ##  计算梯度
            optim.step()  ##  更新参数

            #记录损失值
            if(epochi%10==0):
                loss_d = {}
                loss_d["epoch"] = epochi
                loss_d["songnum"] = listi+1
                loss_d["l1_loss"] = loss_l1.item()
                logging.info("*****专业歌曲损失%s*****",loss_d)

        logging.info("*****第%d次epoch的损失为%f*****",epochi,sumloss)



        #推理阶段
        with torch.no_grad():
            if(epochi%5==0):
                logging.info("*****当前是推理阶段第%d次epoch*****",epochi)
                for listi in range(args.datai):
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

                        pref0=pref0.detach().cpu().reshape(pref0.shape[1]).numpy()

                        pref0 = pref0.astype(np.double)

                        out_path=args.save+"/tmp/"+str(listi+1)+"num_"+str(listj+1)+".wav"
                        y_h = pw.synthesize(pref0, amasp, amaah, args.fs, pw.default_frame_period)
                        # librosa.output.write_wav('result/y_harvest_with_f0_refinement.wav', y_h, fs)
                        sf.write(out_path, y_h, args.fs)
            

if __name__=="__main__":
    main()







