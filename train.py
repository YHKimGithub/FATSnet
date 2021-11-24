import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.utils.data as data

import sys
import os
import os.path
import h5py
import numpy as np
import argparse

from thumos_dataset_precalc_feature import thumos_dataset,thumos_dataset_eval
from model import model
from average_precision import calibrated_ap, ap

feature_rgb_dir='data/rgb_feature_valid_interval5.h5'
feature_flow_dir='data/flow_feature_valid_interval5.h5'
label_dir = 'data/validation_frame.txt'
feature_rgb_dir_test='data/rgb_feature_test_interval5.h5'
feature_flow_dir_test='data/flow_feature_test_interval5.h5'
label_dir_test = 'data/test_frame.txt'
label_videolevel_test = 'data/test_video_level_label.txt'

gpu_id=None
mode = 'train'
batch_size=32
frame_interval=5
frame_length=32
gen_feature_len = 12

workers=0
feature_size=2048
label_num = 21
eval_after_epoch = 50
num_epochs=10000000
learning_rate=5e-5

unit_epoch=10

load_dir=None

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gen_feature_len', type=int, default=12,
                    help='length of generated feature')

args = parser.parse_args()
gen_feature_len=args.gen_feature_len
print(gen_feature_len)

start_epoch=0
model = model(feature_size, label_num)
if(load_dir):
    checkpoint = torch.load(load_dir)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_dir))
    
model.cuda(gpu_id)

if(mode=='train'):
    dataset = thumos_dataset(feature_rgb_dir, feature_flow_dir, label_dir, frame_length*2, frame_interval)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers)
        
    BCE_criterion = nn.BCEWithLogitsLoss()
    MSE_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 70)
        
    max_map = 0.0
    save_flag=False 
    for epoch in range(start_epoch,num_epochs):
        sum_first_loss = 0
        sum_second_loss = 0
        sum_action_loss = 0
        for i, sample_batched in enumerate(dataloader):
            X=sample_batched['data']
            X=Variable(X)
            X=X.cuda(gpu_id)
            Y=sample_batched['label']
            Y= Y.cuda(gpu_id)
            
            prev_prob = torch.zeros(batch_size, label_num).cuda()
            for seg_length in range(1, frame_length/2):
                optimizer.zero_grad()
                first_half = X[:,0:seg_length,:]
                second_half = X[:,frame_length-seg_length:frame_length,:]
                
                if(gen_feature_len >0):
                    h=torch.zeros(first_half.size(0), 4096, requires_grad = True).cuda()
                    for i in range(0, first_half.size(1)):
                        feature, h = model.decoder_forward(first_half[:,i,:], h)
                        
                    sum_loss=0
                    gen_features = [feature]
                    for i in range(0, gen_feature_len-1):
                        feature, h = model.decoder_forward(feature,h)
                        gen_features.append(feature)
                        
                    h=torch.zeros(first_half.size(0),4096, requires_grad=True).cuda()
                    for i in range(len(gen_features)-1, -1, -1):
                        feature, h = model.decoder_backward(gen_features[i], h)
                    
                    for i in range(first_half.size(1)-1, -1, -1):
                        loss = MSE_criterion(feature, first_half[:,i,:])
                        sum_loss += loss
                        feature, h = model.decoder_backward(feature,h)
                    sum_loss.backward()
                    sum_first_loss += sum_loss.item()/first_half.size(1)
                    
                    h=torch.zeros(second_half.size(0), 4096, requires_grad = True).cuda()
                    for i in range(second_half.size(1)-1, -1, -1):
                        feature, h = model.decoder_backward(second_half[:,i,:], h)
                        
                    sum_loss=0
                    gen_features = [feature]
                    for i in range(0, gen_feature_len-1):
                        feature, h = model.decoder_backward(feature,h)
                        gen_features.append(feature)
                        
                    h=torch.zeros(second_half.size(0),4096, requires_grad=True).cuda()
                    for i in range(len(gen_features)-1, -1, -1):
                        feature, h = model.decoder_forward(gen_features[i], h)
                    
                    for i in range(0, second_half.size(1)):
                        loss = MSE_criterion(feature, second_half[:,i,:])
                        sum_loss += loss
                        feature, h = model.decoder_forward(feature,h)
                    sum_loss.backward()
                    sum_second_loss += sum_loss.item()/second_half.size(1)
                
                pred= model(first_half, prev_prob, gen_feature_len)
                prev_prob = pred.detach()
                action_loss=BCE_criterion(pred,Y[:,seg_length-1,:])
                sum_action_loss += action_loss.item()
                action_loss.backward()
                
                optimizer.step()
                
        sum_first_loss = sum_first_loss/(len(dataset)*(frame_length-gen_feature_len))
        sum_second_loss = sum_second_loss/(len(dataset)*(frame_length-gen_feature_len))
        sum_action_loss = sum_action_loss/(len(dataset)*(frame_length-gen_feature_len))
        print('\nEpoch [%d/%d] first_Loss: %.4f second_Loss: %.4f action_Loss: %.4f'
              % (epoch + 1, num_epochs, sum_first_loss, sum_second_loss, sum_action_loss))
        scheduler.step()       
        
        if (epoch + 1) % unit_epoch == 0 and  (eval_after_epoch <= epoch):
            model.eval()
            preds =[]
            labels =[]
            processed_video=[]
            with open(label_dir_test, "r") as fp:
                for string in fp:
                    string = string.split()
                    if(string[0] in processed_video):
                        continue
                    processed_video.append(string[0])
                    
                    vidfile = string[0]
                    #print(vidfile)
                    sys.stdout.write('validation - %s\r'% (vidfile)) 
                    val_dataset = thumos_dataset_eval(feature_rgb_dir_test,feature_flow_dir_test, vidfile, label_dir_test, frame_length, frame_interval, gen_feature_len)
                    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers)
                    
                    duration = val_dataset.label_seq.size(0)    
                    
                    pred = np.zeros((duration,label_num), dtype=np.float32)
                    pred[:,0]=1.0
                    label = val_dataset.label_seq.numpy()
                    label[:, 0:label_num] = np.greater(label, 0.5)
                    
                    with torch.no_grad():     
                        for i, sample_batched in enumerate(val_dataloader):
                            X=sample_batched['data']
                            X=Variable(X)
                            X=X.cuda(gpu_id)
                            idx = sample_batched['idx']
                            prev_prob = torch.zeros(batch_size, label_num).cuda()
                            for seg_length in range(1, frame_length+1):
                                seg = X[:,0:seg_length,:]
                                
                                y_pred = model(seg, prev_prob, gen_feature_len)
                                prev_prob = y_pred.detach()
                                y_pred = nn.Softmax(dim=1)(y_pred)
                                y_pred = y_pred.cpu()
                                for batch in range(0, y_pred.size(0)):
                                    if(idx[batch]+seg_length-1 <duration):
                                        pred[idx[batch]+seg_length-1,:]=y_pred[batch,:]
                    preds.append(pred)
                    labels.append(label)
            
            preds = np.vstack(preds)
            labels = np.vstack(labels)

            AP = ap(labels,preds)
            mAP = np.average(AP[1:])
            print('Average Precisions:')
            print(AP)
            print('mAP:')
            print(mAP)
            if(mAP > max_map):
                max_map=mAP
                save_flag=True
            model.train()
                
            if save_flag:
                save_flag=False
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                  }, 'checkpoints/ckp-genlen%d-ep%d-loss%.4f-map-%.4f.pt'%(gen_feature_len,epoch,sum_action_loss, max_map))
                    
