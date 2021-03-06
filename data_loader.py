import logging
import os
import math
import random
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils as utils

logger = logging.getLogger(__name__)


def data_loader(args, path, batch_size, shuffle=True):
    trn, val, tst = getData(
        data_dir=path,
        hist_len=args.hist_len,
        fut_len=args.fut_len,
        min_ped=args.min_ped,
        delim=args.delim,
        untracked_ratio=args.untracked_ratio)
    print('data processed!')

    loader_trn = DataLoader(
        TrajDataset_loc(trn, args.hist_len),
        batch_size=batch_size[0],
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=collate_loc)
    print('trn!')

    loader_val = DataLoader(
        TrajDataset_trj(val),
        batch_size=batch_size[1],
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=collate_trj)
    print('val!')

    loader_tst = DataLoader(
        TrajDataset_trj(tst),
        batch_size=batch_size[2],
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=collate_trj)
    print('tst!')

    return loader_trn, loader_val, loader_tst


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            if "?" in line: continue
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def getData(data_dir, hist_len=8, fut_len=8, min_ped=1, delim='\t', untracked_ratio=0.5): # threshold=0.002,
    """
    Args:
    - data_dir: Directory containing dataset files in the format
    <frame_id> <ped_id> <x> <y>
    - hist_len: Number of time-steps in input trajectories
    - fut_len: Number of time-steps in output trajectories
    - skip: Number of frames to skip while making the dataset
    - threshold: Minimum error to be considered for non linear traj
    when using a linear predictor
    - min_ped: Minimum number of pedestrians that should be in a seqeunce
    - delim: Delimiter in the dataset files
    """

    all_files = [os.path.join(data_dir, _fn) for _fn in os.listdir(data_dir)]
    seq_len = hist_len+fut_len
    traj, traj_len = [], []
    print('trj')
    for path in all_files:
        print(path)
        if path[-3:] != "txt": continue
        data = read_file(path, delim)
        data = np.around(data, decimals=3)

        frame_id = np.unique(data[:,0])
        num_scene = len(frame_id) - seq_len + 1

        for scene_i in range(num_scene):
            untracked = 0
            data_scene, data_scene_raw = [], data[ np.isin( data[:,0], frame_id[range(scene_i,scene_i+seq_len)] ) ]
            peds = np.unique(data_scene_raw[:,1])
            for i, ped_id in enumerate(peds):
                data_ped = data_scene_raw[data_scene_raw[:, 1]==ped_id]
                if len(data_ped)==seq_len:
                    data_scene.append(data_ped[:,2:].T)
                else:
                    untracked+=1
            if untracked>untracked_ratio*len(peds):
                continue
            num_peds = len(data_scene)
            if num_peds>=min_ped:
                traj_len.append(len(data_scene))
                traj.append(torch.tensor(data_scene).float())

    seed = 11 # random.randint(0,1000)
    random.Random(seed).shuffle(traj)
    random.Random(seed).shuffle(traj_len)
    N = len(traj)
    N0, N1 = int(N*0.8), int(N*0.9)
    print(N,N0,N1)
    return ( (traj[:N0], traj_len[:N0]), (traj[N0:N1], traj_len[N0:N1]), (traj[N1:], traj_len[N1:]) )


class TrajDataset_trj(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data): # threshold=0.002,
        super().__init__()
        self.traj, self.traj_len = data

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, i):
        return (self.traj[i], self.traj_len[i])




class TrajDataset_loc(Dataset):
    def __init__(self, data, hist_len=8):
        super().__init__()
        self.targ_hist, self.cont_hist, self.targ_nextLoc, self.cont_len = [], [], [], []

        traj, traj_len = data
        N, T = len(traj), traj[0].shape[2]
        for scene_i in range(N):
            num_peds = traj_len[scene_i]
            data_scene_full, _idx = traj[scene_i].numpy().swapaxes(1,2), np.arange(num_peds)
            for t in range(1): # (T-hist_len-1):
                data_scene = data_scene_full[:,t:t+hist_len+1]
                self.cont_len = np.append(self.cont_len,[num_peds-1]*num_peds)
                for i in range(num_peds):
                    dt = data_scene - data_scene[i,hist_len-1]
                    tht = np.pi/2+np.arctan2(*dt[i,hist_len-2])
                    Rot = np.array([[np.cos(tht),-np.sin(tht)],[np.sin(tht),np.cos(tht)]])
                    dt = (Rot@dt.reshape(-1,2).T).reshape(2,-1,hist_len+1).swapaxes(0,1)
                    dt = torch.tensor(dt).float()
                    self.targ_nextLoc.append(dt[i,:,-1])
                    self.targ_hist.append(dt[i,:,:-1])
                    self.cont_hist.append(dt[_idx!=i,:,:-1])

    def __len__(self):
        return len(self.targ_hist)

    def __getitem__(self, i):
        return (self.targ_hist[i], self.cont_hist[i], self.cont_len[i], self.targ_nextLoc[i])



def collate_trj(data):
    traj, traj_len = zip(*data)
    end_idx = np.append(0,np.cumsum(traj_len).astype("int"))
    return (torch.cat(traj,dim=0), end_idx)

def collate_loc(data):
    targ_hist, cont_hist, cont_len, targ_nextLoc = zip(*data)
    end_idx = np.append(0,np.cumsum(cont_len).astype("int"))
    targ_hist = torch.stack(targ_hist)
    cont_hist = torch.cat(cont_hist,dim=0)
    targ_nextLoc = torch.stack(targ_nextLoc)
    out = (targ_hist, cont_hist, end_idx, targ_nextLoc)
    return out



#
