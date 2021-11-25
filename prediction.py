import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.utils.data as data

import os
import os.path
import h5py
import numpy as np

from thumos_dataset_precalc_feature import thumos_dataset_eval
from model import model

gpu_id=None
mode = 'test'
batch_size=1
frame_interval=5
frame_length=32
gen_feature_len = 12

workers=0
feature_size=2048
label_num = 21

infer_steps=4

feature_rgb_dir_test='data/rgb_feature_test_interval5.h5'
feature_flow_dir_test='data/flow_feature_test_interval5.h5'
label_dir_test = 'data/test_frame.txt'
label_videolevel_test = 'data/test_video_level_label.txt'

outfile_path='prediction.h5'

load_dir=None
load_dir='./checkpoints/ckp-genlen12-ep205-best.pt'

model = model(feature_size, label_num)
model.cuda(gpu_id)

if(load_dir):
    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_dir))

if(mode=='test'):
    outfile = h5py.File(outfile_path, 'w')
    
    processed_video=[]
    with open(label_videolevel_test, "r") as fp:
        for string in fp:
            string = string.split()
            if(string[0] in processed_video):
                continue
            processed_video.append(string[0])
            
            vidfile = string[0]
            print(vidfile)
            val_dataset = thumos_dataset_eval(feature_rgb_dir_test,feature_flow_dir_test, vidfile, label_dir_test, frame_length, frame_interval, gen_feature_len)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=workers)
            
            duration = val_dataset.label_seq.size(0)    
            print(val_dataset.label_seq.size())
            dset_label = outfile.create_dataset(string[0]+'/label', (duration,label_num), maxshape=(duration,label_num), chunks=True, dtype=np.float32)
            dset_label[:,:] = val_dataset.label_seq.numpy()
            
            dset_pred = outfile.create_dataset(string[0]+'/pred', (duration,label_num), maxshape=(duration,label_num), chunks=True, dtype=np.float32)
            dset_pred[:,0]=1.0
            dset_weight = outfile.create_dataset(string[0]+'/weight', (duration,2), maxshape=(duration,2), chunks=True, dtype=np.float32)
            dset_weight[:,1]=1.0
            model.eval()
            with torch.no_grad():   
                seg = []
                seg_h = []
                seg_decode_h = []
                for i in range(0, infer_steps):
                    seg.append(-i * frame_length/infer_steps)
                    seg_h.append( torch.zeros(batch_size, 4096, requires_grad = True).cuda() )
                    seg_decode_h.append( torch.zeros(batch_size, 4096, requires_grad = True).cuda() )
                
                for i, sample_batched in enumerate(val_dataloader):
                    X=sample_batched['data']
                    X=Variable(X)
                    X=X.cuda(gpu_id)
                    idx = sample_batched['idx']
                    prev_prob = torch.zeros(batch_size, label_num).cuda()
                    
                    for seg_idx in range(0, frame_length):
                        current_feature = X[:,seg_idx,:]
                        
                        infer_cnt = 0
                        for infer_idx in range(0,infer_steps):
                            if(seg[infer_idx] >= 0):
                                infer_cnt += 1
                                if(seg[infer_idx] == frame_length ):
                                    seg[infer_idx] = 0
                                    seg_h[infer_idx].zero_()
                                    seg_decode_h[infer_idx].zero_()
                                
                                y_pred, seg_decode_h[infer_idx], seg_h[infer_idx], weight = model.forward_unroll(current_feature, seg_decode_h[infer_idx], seg_h[infer_idx], prev_prob, gen_feature_len)
                                y_pred = nn.Softmax(dim=1)(y_pred)
                                y_pred = y_pred.cpu().numpy()
                                weight = weight.cpu().numpy()
                                
                                for batch in range(0, y_pred.shape[0]):
                                    if(idx[batch]+seg_idx <duration):
                                        dset_pred[idx[batch]+seg_idx,:]+=y_pred[batch,:] 
                                        dset_weight[idx[batch]+seg_idx,:]=weight[batch,:]    
                                        
                            seg[infer_idx] += 1
                        for batch in range(0, X.size(0)):
                            if(idx[batch]+seg_idx <duration):
                                dset_pred[idx[batch]+seg_idx,:]/=infer_cnt
                                prev_prob = torch.from_numpy(dset_pred[idx[0]+seg_idx,:]).view(1,-1).cuda()
                        
                                
