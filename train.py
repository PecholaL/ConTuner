from net import DiffNet
from getmel import *
from diffusion import GaussianDiffusion

import argparse
import torch
import logging
import sys
import soundfile
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("ConTuner")
parser.add_argument(
    "--save",
    type=str,
    default="/home/Coding/ConTuner/NlpVoice/ConTuner/result/cryresult",
    help="experiment name",
)
parser.add_argument(
    "--hidden_size1", type=int, default=256, help="the size of hidden cell"
)
parser.add_argument("--audio_num_mel_bins", type=int, default=80)
parser.add_argument(
    "--timesteps", type=int, default=100, help="the steps of the diffusion"
)
parser.add_argument("--timescale", type=int, default=1)
parser.add_argument("--diff_loss_type", type=str, default="l1")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--train_epoch", type=int, default=1000000)

args = parser.parse_args()


log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def write_line2log(log_dict: dict, filedir, isprint: True):
    strp = ""
    with open(filedir, "a", enCoding="utf-8") as f:
        for key, value in log_dict.items():
            witem = "{}".format(key) + ":{},".format(value)
            strp += witem
        f.write(strp)
        f.write("\n")
    if isprint:
        print(strp)
    pass


def main():
    ji = r"/home/Coding/data/cry.wav"

    mel, _ = get_spectrograms(ji)
    mel = np.array(mel)

    plt.figure(figsize=(16, 8))
    librosa.display.TimeFormatter(lag=True)
    mel_img = librosa.display.specshow(mel, y_axis="mel", x_axis="s")  # , fmax=8000
    plt.title(f"Mel-Spectrogram")
    plt.colorbar(mel_img, format="%+2.0f dB")
    plt.savefig("/home/Coding/ConTuner/NlpVoice/ConTuner/result/cryresult/image1.png")
    plt.close()

    mel = torch.from_numpy(mel)
    mel = mel.to(device)
    print("mel.shape: ", mel.shape)
    logging.info("mel.shape: ", mel.shape)

    denoise_model = DiffNet(80)
    denoise_model = denoise_model.to(device)

    diffusion_model = GaussianDiffusion(
        out_dims=args.audio_num_mel_bins,
        denoise_fn=denoise_model,
        timesteps=args.timesteps,
        time_scale=args.timescale,
        loss_type=args.diff_loss_type,
        spec_min=[],
        spec_max=[],
    )
    diffusion_model = diffusion_model.to(device)

    loss_f = torch.nn.L1Loss().to(device)  ## |X-Y|
    optim = torch.optim.Adam(
        denoise_model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-09,
        weight_decay=1e-8,
    )

    cond = torch.rand(size=(args.batch_size, mel.shape[0], mel.shape[1])).to(device)

    for traini in range(args.epoch):
        input = mel.reshape(
            args.batch_size, 1, mel.shape[0], mel.shape[1]
        )  # [B,1,80,T]
        input = input.to(device)
        print("*****epoch: ", traini, "*****")
        logging.info("*****epoch: %d*****", traini)
        for j in range(args.train_epoch):
            t = torch.randint(
                0, args.timesteps + 1, (args.batch_size,), device=device
            ).long()

            diffusionz = mel.reshape(
                args.batch_size, mel.shape[1], mel.shape[0]
            )  # [B,T_s,80]

            # Diffusion
            x_t = diffusion_model.diffuse_fn(
                diffusionz, t
            )  # [B,1,80,T]   # [B, T_s, 80]

            x_0_pred = denoise_model(x_t, t.reshape(-1, 1), cond)  # [B,1,80,T]

            loss_l1 = loss_f(x_0_pred, input)

            optim.zero_grad()
            loss_l1.backward()
            optim.step()

            if traini % 1 == 0:
                loss_d = {}
                loss_d["epoch"] = traini
                loss_d["step"] = j
                loss_d["l1_loss"] = loss_l1.item()
                write_line2log(
                    loss_d,
                    "/home/Coding/ConTuner/NlpVoice/ConTuner/result/cryresult/train_log.txt",
                    True,
                )

        """ Inference
        """
        if traini % 10 == 0:
            t = args.timesteps
            x = x_t
            for i in tqdm(
                reversed(range(0, t)), desc="ProDiff sample time step", total=t
            ):
                x = diffusion_model.p_sample(
                    x,
                    torch.full((args.batch_size,), i, device=device, dtype=torch.long),
                    cond,
                )  # x(mel), t, condition(phoneme)

            tmp = x.reshape(x.shape[2], x.shape[3])
            tmp = tmp.cpu().numpy()

            plt.figure(figsize=(16, 8))
            librosa.display.TimeFormatter(lag=True)
            mel_img = librosa.display.specshow(
                tmp, y_axis="mel", x_axis="s"
            )  # , fmax=8000
            plt.title(f"Mel-Spectrogram")
            plt.colorbar(mel_img, format="%+2.0f dB")
            plt.savefig(
                "/home/Coding/ConTuner/NlpVoice/ConTuner/result/cryresult/imageepoch"
                + str(traini)
                + ".png"
            )
            plt.close()

            wav1 = melspectrogram2wav(tmp.T)  # input size : (frames ,ndim)
            sr = 16000
            outputfile = (
                "/home/Coding/ConTuner/NlpVoice/ConTuner/result/cryresult/stdioepoch"
                + str(traini)
                + ".wav"
            )
            soundfile.write(outputfile, wav1, sr)
    pass


if __name__ == "__main__":
    main()
