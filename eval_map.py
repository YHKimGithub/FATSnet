import os
import numpy as np
import h5py
from average_precision import ap
import argparse

num_classes=21
label_file = 'data/test_frame.txt'
pred_path='prediction.h5'

parser = argparse.ArgumentParser(description='prediction file path.')
parser.add_argument('--pred_path', type=str, default='prediction.h5',
                    help='prediction file path')

args = parser.parse_args()
pred_path=args.pred_path
print(pred_path)

f = h5py.File(pred_path,'r')

processed_video=[]

preds =[]
labels =[]
with open(label_file, "r") as fp:
    for string in fp:
        string = string.split()
        if(string[0] in processed_video):
            continue
        processed_video.append(string[0])
        
        pred = f[string[0]+'/pred']
        label = f[string[0]+'/label']
        
        pred_reshape = np.zeros((pred.shape[0],pred.shape[1]), dtype=np.float32)
        label_bool = np.zeros((pred.shape[0],pred.shape[1]), dtype=np.bool)
        pred_reshape[:, 0:pred.shape[1]] = pred[:,:]
        label_bool[:, 0:pred.shape[1]] = np.greater(label, 0.5)
               
        labels.append(label_bool)
        preds.append(pred_reshape)
        
preds = np.vstack(preds)
labels = np.vstack(labels)

ap = ap(labels,preds)

print('Average Precisions:')
print(ap)
print('mAP:')
print(np.average(ap[1:]))

