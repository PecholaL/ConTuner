import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


#设置一些超参数
parser = argparse.ArgumentParser("DiffBeautifier")
parser.add_argument('--hidden_size', type=int, default=80)
parser.add_argument('--residual_layers', type=int, default=20)
parser.add_argument('--residual_channels', type=int, default=256)
parser.add_argument('--dilation_cycle_length', type=int, default=1)
args = parser.parse_args()


# 实现Mish激活函数
# used as class:
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))






class AttrDict(dict):
    def __init__(self, *args, **kwargs):  #不确定变量个数的tube和dict
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#对一维卷积做了一些初始化的工作
def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

#wavenet中间那块
class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)  #
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        # print("****")
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)  
        # print("*******")
        y = x + diffusion_step
      
        y=y[:,0]  #[B,1,residual_channel,T]->[B,residual_channel,T]

        # print("y",y.shape)
        # print("***",self.dilated_conv(y).shape,"&&&",conditioner.shape)
        y = self.dilated_conv(y) + conditioner  #将三个部分糅合在一起

        gate, filter = torch.chunk(y, 2, dim=1)  #在第一维上拆成两份
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


#降噪器的整体结构
class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=args.hidden_size,    #256
            residual_layers=args.residual_layers,    #20 
            residual_channels=args.residual_channels,     #256
            dilation_cycle_length=args.dilation_cycle_length,    #1，这个不知道干啥用的
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)  #x做的卷积
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)  #t做的位置编码
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),     #Mish()是一个激活函数
            nn.Linear(dim * 4, dim)
        )   # t做的前向连接

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]   #[B,M,T]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)


        diffusion_step = self.diffusion_embedding(diffusion_step)   #传入时间步数t，得到的diffusion_step维度为[B,1,params.residual_channels]
        diffusion_step = self.mlp(diffusion_step)


        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]  六
        return x[:, None, :, :]    #[B,1,80,T]由回到了最初的一个输入的x的维度
        return x

#音高预测器的整体结构
class PitchNet(nn.Module):
    def __init__(self,in_dim=513, out_dim=256, kernel=5, n_layers=3, strides=None):
        super().__init__()

        # self.in_linear=nn.Linear(2,513)

        self.in_linear = nn.Sequential(
            nn.Linear(1,  16),
            Mish(),     #Mish()是一个激活函数
            nn.Linear(16, 64),
            Mish(),
            nn.Linear(64, 256), 
            Mish(),
            nn.Linear(256 , 513)
        ) 

        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=padding, stride=self.strides[l]),
                nn.ReLU(),
                # nn.Dropout(0.01),
                nn.BatchNorm1d(out_dim)
            ))
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)


        self.mlp = nn.Sequential(
            nn.Linear(out_dim,  out_dim // 4),
            Mish(),     #Mish()是一个激活函数
            nn.Linear(out_dim//4, out_dim//16),
            Mish(),
            nn.Linear(out_dim//16, out_dim//64), 
            Mish(),
            nn.Linear(out_dim//64 , 1)
        )   
        self.out_proj = nn.Linear(out_dim,out_dim)

    def forward(self,sp_h,midi):
        '''
        sp_h:[B,M,513]
        midi:[B,M,1]
        output:[B,M,]  一维向量
        '''
        # print("&&&&&",sp_h.shape)

        # print(sp_h.shape,midi.shape)

        # height1=sp_h.shape[1]
        # weight=sp_h.shape[2]
        # height2=midi.shape[1]

        # print("******",midi.dtype)

        midi=self.in_linear(midi) #[B,n,513]

        # print("&&&&&&")

        # for i in range()

        # print("&&&&&&",sp_h)


        # for i in range(midi.shape[0]):

        #     tmp=midi[:,i,:]

        #     # print("*****",tmp)

        #     tmp=tmp.reshape(tmp.shape[0],1,tmp.shape[1])
        #     tmp1=tmp*0.1
        #     tmp2=tmp1+sp_h
        #     sp_h=tmp2


        x=torch.cat([midi,sp_h],dim=1)  #进行拼接

        x=sp_h.transpose(1,2)

        
        for i, l in enumerate(self.layers):
            x=l(x)

        x=x.transpose(1,2)
        x=self.mlp(x)
        x=x.reshape(x.shape[0],x.shape[1])

        # print("*****",x.shape)

        return x

class DiffNetCon(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=args.hidden_size,    #256
            residual_layers=args.residual_layers,    #20 
            residual_channels=args.residual_channels,     #256
            dilation_cycle_length=args.dilation_cycle_length,    #1，这个不知道干啥用的
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)  #x做的卷积
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)  #t做的位置编码
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),     #Mish()是一个激活函数
            nn.Linear(dim * 4, dim)
        )   # t做的前向连接

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, 750]->经过长度规整器之后变成[B, M, T]
        :return:
        """

        mel_len=spec.shape[3]

        x = spec[:, 0]   #[B,M,T]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)


        diffusion_step = self.diffusion_embedding(diffusion_step)   #传入时间步数t，得到的diffusion_step维度为[B,1,params.residual_channels]
        diffusion_step = self.mlp(diffusion_step)

        #进行con的长度规整
        expand_factor=mel_len//750

        regulator=Length_Regulator(expand_factor)

        cond=cond.transpose(1,2)

        cond=regulator(cond,mel_len)

        cond=cond.transpose(1,2)

        # print("netcon",cond.shape)

        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]  六
        return x[:, None, :, :]    #[B,1,80,T]由回到了最初的一个输入的x的维度
        return x



class Length_Regulator(nn.Module):
    def __init__(self,expand_factor):
        '''
        Length_Regulator 简单扩充
        输入 文本编码 [B,Text_len,D],mel_len
        输出 melspec 编码 【B，mel_len,D】
        '''
        super().__init__()
        # 下面这个变量记录每个文字要扩充的melspec长度。
        self.expand_factor = expand_factor

    def forward(self,text_memory,mel_len):
        '''
        这里这个函数比较特殊，我们根据一个固定的数值来进行 expand !
        而不是像 fastspeech2那样，根据一个数组（predicted duration）来进行 扩充！
        :param text_memory: [B,Text_len,D]
        :param mel_len: a number,target mel len
        :return:  [B,mel len ,D]
        '''

        ## 把 text_memory 中的 text len 直接倍数 扩增到 mel len!

        mel_chunks = []
        text_len = text_memory.shape[1] ## "七百八十九 = 5
        for t in range(text_len):
            ## 取出第 t 个时间步的 文本向量，并重复factor个时间步
            t_vec = text_memory[:,t,:].unsqueeze(1) ## [B,1,D]
            ## 将 t_vec  重复 self.expand_factor 次
            t_vec = t_vec.repeat(1,self.expand_factor,1)  ## [B,1,D] --->[B,self.expand_factor,D]
            ## 此时 可以认为第t个文本已经扩充到了 其对应的 melspec的长度。
            mel_chunks.append(t_vec)

        ## 下面 的代码，是一个输出长度的修正。
        ## [B,self.expand_factor * text len,D] --->[B,melspec len,512]

        ## 在时间维度上，拼接所有时间帧
        mel_chunks = torch.cat(mel_chunks,dim=1)
        B,cat_mel_len , D = mel_chunks.shape
        ## 确认其长度与给定的目标melspec一致
        ##
        if cat_mel_len < mel_len:
            ## 此时 需要padding,拼接一个 全零张量
            pad_t = torch.zeros((B,mel_len-cat_mel_len,D),device=text_memory.device)
            mel_chunks = torch.cat([mel_chunks,pad_t],dim=1)
        else:
            mel_chunks = mel_chunks[:,:mel_len,:]
        return mel_chunks ## [B,melspec,512]
