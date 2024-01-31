import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

parser = argparse.ArgumentParser("ConTuner")
parser.add_argument("--hidden_size", type=int, default=80)
parser.add_argument("--residual_layers", type=int, default=20)
parser.add_argument("--residual_channels", type=int, default=256)
parser.add_argument("--dilation_cycle_length", type=int, default=1)
args = parser.parse_args()


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
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


# initialization for Conv1d
def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(residual_channels, residual_channels)  #
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        # print("****")
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        # print("*******")
        y = x + diffusion_step

        y = y[:, 0]  # [B,1,residual_channel,T]->[B,residual_channel,T]

        # print("y",y.shape)
        # print("***",self.dilated_conv(y).shape,"&&&",conditioner.shape)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


# Denoiser
class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=args.hidden_size,  # 256
            residual_layers=args.residual_layers,  # 20
            residual_channels=args.residual_channels,  # 256
            dilation_cycle_length=args.dilation_cycle_length,  # 1
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    params.encoder_hidden,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]  # [B,M,T]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(
            diffusion_step
        )  # diffusion_step shape [B,1,params.residual_channels]
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]  六
        return x[:, None, :, :]


# Pitch Predictor
class PitchNet(nn.Module):
    def __init__(self, in_dim=513, out_dim=256, kernel=5, n_layers=3, strides=None):
        super().__init__()

        # self.in_linear=nn.Linear(2,513)

        self.in_linear = nn.Sequential(
            nn.Linear(1, 16),
            Mish(),
            nn.Linear(16, 64),
            Mish(),
            nn.Linear(64, 256),
            Mish(),
            nn.Linear(256, 513),
        )

        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        out_dim,
                        kernel_size=kernel,
                        padding=padding,
                        stride=self.strides[l],
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                )
            )
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4),
            Mish(),
            nn.Linear(out_dim // 4, out_dim // 16),
            Mish(),
            nn.Linear(out_dim // 16, out_dim // 64),
            Mish(),
            nn.Linear(out_dim // 64, 1),
        )
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, sp_h, midi):
        """
        sp_h:[B,M,513]
        midi:[B,M,1]
        output:[B,M,]
        """
        midi = self.in_linear(midi)  # [B,n,513]

        x = torch.cat([midi, sp_h], dim=1)

        x = sp_h.transpose(1, 2)

        for _, l in enumerate(self.layers):
            x = l(x)

        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.reshape(x.shape[0], x.shape[1])

        return x


class ExpressivenessEnhancer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class DiffNetCon(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=args.hidden_size,  # 256
            residual_layers=args.residual_layers,  # 20
            residual_channels=args.residual_channels,  # 256
            dilation_cycle_length=args.dilation_cycle_length,  # 1
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    params.encoder_hidden,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, 750]->[B, M, T]
        :return:
        """

        mel_len = spec.shape[3]

        x = spec[:, 0]  # [B,M,T]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        expand_factor = mel_len // 750
        regulator = LengthRegulator(expand_factor)

        cond = cond.transpose(1, 2)
        cond = regulator(cond, mel_len)
        cond = cond.transpose(1, 2)

        skip = []
        for _, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x[:, None, :, :]


class LengthRegulator(nn.Module):
    def __init__(self, expand_factor):
        """Length_Regulator"""
        super().__init__()
        self.expand_factor = expand_factor

    def forward(self, text_memory, mel_len):
        """
        param text_memory: [B,Text_len,D]
        param mel_len: a number,target mel len
        return:  [B,mel len ,D]
        """

        mel_chunks = []
        text_len = text_memory.shape[1]
        for t in range(text_len):
            t_vec = text_memory[:, t, :].unsqueeze(1)  ## [B,1,D]
            t_vec = t_vec.repeat(
                1, self.expand_factor, 1
            )  ## [B,1,D] --->[B,self.expand_factor,D]
            mel_chunks.append(t_vec)

        ## [B,self.expand_factor * text len,D] --->[B,melspec len,512]

        mel_chunks = torch.cat(mel_chunks, dim=1)
        B, cat_mel_len, D = mel_chunks.shape

        if cat_mel_len < mel_len:
            pad_t = torch.zeros(
                (B, mel_len - cat_mel_len, D), device=text_memory.device
            )
            mel_chunks = torch.cat([mel_chunks, pad_t], dim=1)
        else:
            mel_chunks = mel_chunks[:, :mel_len, :]
        return mel_chunks  ## [B,melspec,512]
