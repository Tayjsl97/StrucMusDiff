import torch
import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
import pypianoroll,muspy
from muspy import (
    Music,
    Note,
    Tempo,
    Track,
)
from skimage.metrics import structural_similarity as SSIM
np.set_printoptions(threshold=np.inf)


def nested_map(struct, map_fn):
    """This is for trasfering into cuda device"""
    if isinstance(struct, tuple):
        return tuple(nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def compute_sim_metrics(img1,img2,real_sim_v):
    img1 = unnormalize(img1)
    img2 = unnormalize(img2)
    img1 = float2Interger(img1).astype(int)
    img2 = float2Interger(img2).astype(int)
    gen_sim_v = SSIM(img1, img2, win_size=13, data_range=127)
    print("gen_sim_V: ", round(gen_sim_v, 2), ", real_sim_v: ", round(real_sim_v, 2))
    return abs(gen_sim_v-real_sim_v)


def show_image(img: torch.Tensor, title="", cmap="Greens"):
    """Helper function to display an image"""
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    assert img.ndim == 3
    if img.shape[0]==1:
        img = img[0]
    else:
        img = img[0] + img[1]
    if title.find("similar_phrase")!=-1:
        plt.rcParams['figure.figsize'] = (10, 7)
    else:
        plt.rcParams['figure.figsize'] = (20, 7)
    fig, axs = plt.subplots()
    img = pypianoroll.plot_pianoroll(axs, img, grid_axis='x', \
                                     resolution=4, cmap=cmap, preset='full')
    plt.tight_layout()
    if title!="":
        plt.savefig(title)
    plt.close()


def float2Interger(pr):
    new_pr=np.zeros_like(pr)
    for i in range(pr.shape[0]): #pitch
        for j in range(pr.shape[1]): # time
            if pr[i][j]>=125:
                new_pr[i][j] = 128
            elif j>0 and abs(pr[i][j]-pr[i][j-1])>30 and pr[i][j]>110:
                new_pr[i][j] = 128
            elif j>0 and pr[i][j-1]==128 and pr[i][j]>110:
                new_pr[i][j] = 128
            elif pr[i][j]>5:
                new_pr[i][j]=np.round(pr[i][j])
    return new_pr


def print_pr(pr):
    if pr.ndim == 4:
        pr = pr[0][0]
    elif pr.ndim == 3:
        pr = pr[0]
    pr=pr.transpose()
    pr=unnormalize(pr)
    pr=float2Interger(pr).astype(int)
    print("pr_shape: ",pr.shape)
    for i in range(pr.shape[0]):
        print("No. ",i,": ",pr[i])


def pr_to_midi_file(pr, fpath):
    if pr.ndim == 4:
        pr = pr[0][0]
    elif pr.ndim == 3:
        pr = pr[0]
    pr=pr.transpose()
    pr=unnormalize(pr)
    pr=float2Interger(pr).astype(int)
    notes=[]
    for i in range(pr.shape[0]): #pitch
        print(i,pr[i])
        j = 0
        while j < pr.shape[1]:  # time
            if pr[i][j] == 0:
                j = j + 1
                continue
            if pr[i][j]==128:
                j+=1
            else:
                end=pr.shape[1]
                for k in range(j+1,end):
                    if pr[i][k]!=128:
                        end=k
                        break
                notes.append(Note(
                    time=j,
                    pitch=i,
                    duration=end - j,
                    velocity=pr[i][j]
                ))
                j=end
    track = Track(program=0, is_drum=False, notes=notes)
    music = Music(resolution=4, tracks=[track], tempos=[Tempo(time=0, qpm=120)])
    muspy.write_midi(fpath, music)
    notes.sort(key=attrgetter("time", "pitch", "duration", "velocity"))
    notes_list = []
    phrase_num = (notes[-1].time + 1) // 64
    if (notes[-1].time + 1) % 64 != 0:
        phrase_num += 1
    for i in range(phrase_num):
        phrase = []
        for j in notes:
            if j.time < (i + 1) * 64 and j.time >= (i) * 64:
                phrase.append(j)
            if j.time >= (i + 1) * 64:
                break
        notes_list.append(phrase)
    return notes_list


def normalize(x):
    mean = 4.3653
    std = 22.6885
    x -= mean
    x /= (std+1e-8)
    return x


def unnormalize(x):
    mean = 4.3653
    std = 22.6885
    x *= (std+1e-8)
    x += mean
    return x
