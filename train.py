# run
# CUDA_VISIBLE_DEVICES="1" python -i scnn/pg.py

import argparse
import gc
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import data_loader
# import data_loader0

import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep

import model as model
import utils as utils

import datetime
_TIME = str(datetime.datetime.now())[5:19]

torch.backends.cudnn.benchmark = True

from Arguments import Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--token', default='', type=str)
args_set = parser.parse_args()
args = Arguments(args_set.token)

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

''' Load data '''
npy_path = '/home/dzhao/proj/scnn/datasets/'+args.dataset_name+'_loaders.npy'
if args.loadNpy:
    logger.info("loading data from npy ...")
    loader_trn, loader_val, loader_tst = np.load(npy_path,allow_pickle=True)
else:
    logger.info("reading data from txt ...")
    data_path = './data/'+args.dataset_name
    loader_trn, loader_val, loader_tst = data_loader(args, data_path, batch_size=(args.batch_size,args.batch_size_val,args.batch_size_tst))
    # trn_path = utils.get_dset_path(args.dataset_name, 'train')
    # val_path = utils.get_dset_path(args.dataset_name, 'val')
    # tst_path = utils.get_dset_path(args.dataset_name, 'test')
    # loader_trn = data_loader0.data_loader(args, trn_path, fut_loc=True, batch_size=args.batch_size)
    # loader_val = data_loader0.data_loader(args, val_path, fut_loc=False, batch_size=args.batch_size_val)
    # loader_tst = data_loader0.data_loader(args, tst_path, fut_loc=False, batch_size=args.batch_size_tst)
    np.save(npy_path,[loader_trn, loader_val, loader_tst])

num_sample = len(loader_trn.dataset)
iterations_per_epoch = num_sample/args.batch_size
args.n_iteration = min(args.n_iteration,iterations_per_epoch)
logger.info('{} samples in an epoch, {} iterations per epoch'.format(num_sample,iterations_per_epoch))
logger.info('{} epochs;  batch_size: {} '.format(args.n_epoch,args.batch_size))
print("max()"if args.use_max else "sum()")



''' Construct Model '''
gpu, cpu = "cuda:0", "cpu"
predictor = utils.load_model(args.loadModel) if args.loadModel else model.LocPredictor(args).to(gpu)
optimizer = optim.Adam(predictor.parameters(), lr=args.lr)

t0 = time()
err = []
nan_cnt = 0
best_i, min_err = -1, 9999
model_i = 0
# plt.figure(); plt.ion(); plt.show()
for epoch in range(args.n_epoch):
    # predictor.c_conv[0].lock_weights(epoch<8)

    sum_trn_loss = 0.0
    for i, b in enumerate(loader_trn):
        if i>=args.n_iteration: break
        targ_hist, cont_hist, end_idx, targ_nextLoc = b
        loc_pred = predictor.forward(targ_hist.to(gpu), cont_hist.to(gpu), end_idx)
        loss = utils.getLoss(targ_nextLoc, loc_pred.to(cpu))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_trn_loss += loss

        print('\r[%d-%d/%d-%d] %.3f' % (model_i,epoch,args.n_epoch,i,sum_trn_loss/(i+1)),end=' ')
        if torch.isnan(sum_trn_loss): print("NaN!!!");raise Exception('Nan error.')

    # optimizer.defaults['lr'] = utils.update_lr(epoch,optimizer.defaults['lr'])
    t1 = time()-t0
    print("%.2fs/%.2fm %f"%(t1,t1/60, optimizer.defaults['lr']), end=' ')

    if (epoch+1)%args.val_freq==0:
        # utils.plotWeights(predictor.c_conv[0].weight.detach().to(cpu),fn="weights/2layer/"+args.dataset_name+"/7aug"+str(epoch) )
        ade_vd,fde_vd= utils.eval_model(predictor,loader_val,determ=1,n_batch=args.n_batch_val)
        ade_v, fde_v = 0,0 # utils.eval_model(predictor,loader_val,determ=0,n_batch=args.n_batch_val)
        print('v%.3f/%.3f' % (ade_vd,fde_vd), end=' ')
        err.append([ade_vd,fde_vd,ade_v,fde_v])
        err_val = (ade_vd+fde_vd)
        if epoch>0.7*args.n_epoch:
            if err_val<min_err or best_i<0:
                best_i, min_err = len(err)*args.val_freq, err_val
                utils.save_model(predictor,fn=args.dataset_name+"_"+str(best_i)+"_"+args.token)
    print('!')


best_predictor = utils.load_model(fn=args.dataset_name+"_"+str(best_i)+"_"+args.token)
ade_t, fde_t = utils.eval_model(best_predictor,loader_tst,determ=0,n_batch=args.n_batch_tst,repeat=1)
# Err.append([ade_t,fde_t])
# print('tp:%.3f/%.3f' % (ade_t,fde_t))
print('Finished Training',time()-t0)
# utils.write_csv([model_i,*Params[model_i],best_i,ade_t,fde_t],fn=args.dataset_name)
utils.write_csv([model_i,best_i,ade_t,fde_t],fn=args.dataset_name)
#     utils.plot_err(err,ade_t,fde_t,fn=args.dataset_name+str(model_i))
