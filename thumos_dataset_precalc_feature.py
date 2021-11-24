import torch
import torch.utils.data as data

import numpy as np
import os
import os.path
import random
import h5py

num_classes=21

class thumos_dataset(data.Dataset):
    def __init__(self, feature_rgb_dir,feature_flow_dir, label_dir, frame_length, frame_interval):
        self.feature_rgb_dir = feature_rgb_dir
        self.feature_flow_dir = feature_flow_dir
        self.label_dir = label_dir
        self.samples = []
        self.frame_length = frame_length
        self.frame_interval = frame_interval
        self.feature_rgb_file = h5py.File(feature_rgb_dir, 'r')
        self.feature_flow_file = h5py.File(feature_flow_dir, 'r')

        with open(label_dir, "r") as fp:
            for string in fp:
                string = string.split()
                flag = False
                for i, sample in enumerate(self.samples):
                    if ( sample[0] == string[0]) :
                        label_seq = self.samples[i][1]
                        label = torch.zeros(num_classes, dtype=torch.float32)
                        label[int(string[3])]=1
                        
                        st=int(int(string[1])/self.frame_interval)
                        ed=int(int(string[2])/self.frame_interval)+1
                        if(st < label_seq.size(0) and ed < label_seq.size(0)):
                            label_seq[st:ed,0]=0
                            label_seq[st:ed]+=label
                            label_seq=torch.clamp(label_seq, min=0, max=1)                            
                            self.samples[i][1]=label_seq
                        flag = True
                        
                if( not flag ):
                    duration = min(len(self.feature_rgb_file[string[0]]), len(self.feature_flow_file[string[0]]))
                    label_seq = torch.zeros([int(duration), num_classes],dtype=torch.float32)
                    label_seq[:,0]=1
                    
                    label = torch.zeros(num_classes, dtype=torch.float32)
                    label[int(string[3])]=1
                    
                    st=int(int(string[1])/self.frame_interval)
                    ed=int(int(string[2])/self.frame_interval)+1
                    if(st < label_seq.size(0) and ed < label_seq.size(0)):
                        label_seq[st:ed]=label
                        self.samples.append([string[0], label_seq])
        
    def __getitem__(self, index):
        framedata = []
        filename = self.samples[index][0]
        label_seq = self.samples[index][1]
        
        feature_rgb = self.feature_rgb_file[filename]
        feature_flow = self.feature_flow_file[filename]
        duration = min(len(feature_rgb), len(feature_flow))
        feature = np.append(feature_rgb[:duration,:],feature_flow[:duration,:], axis=1)
        
        if(duration < self.frame_length):
            start_frame=0
            frames = feature[0:duration]
            frames = np.append(feature,feature[0:self.frame_length-duration], axis=0)
            label = label_seq[0:duration]
            label = np.append(label,label_seq[0:self.frame_length-duration], axis=0)
        else:
            start_frame = random.randrange(0, duration-self.frame_length)
            
            frames = feature[start_frame:start_frame+self.frame_length]
            label = label_seq[start_frame:start_frame+self.frame_length]
            
        assert(len(frames)==self.frame_length)
        assert(len(label)==self.frame_length)
        return {'data': torch.from_numpy(np.array(frames)), 'label': torch.from_numpy(np.array(label))}
        
    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return self.samples
        
        
class thumos_dataset_eval(data.Dataset):
    def __init__(self, feature_rgb_dir,feature_flow_dir, filename, label_dir, frame_length, frame_interval,gen_feature_len):
        self.feature_rgb_dir = feature_rgb_dir
        self.feature_flow_dir = feature_flow_dir
        self.filename = filename
        self.label_dir = label_dir
        self.samples = []
        self.frame_length = frame_length
        self.frame_interval = frame_interval
        self.gen_feature_len = gen_feature_len
        self.feature_rgb_file = h5py.File(feature_rgb_dir, 'r')
        self.feature_flow_file = h5py.File(feature_flow_dir, 'r')
        
        self.feature_rgb=self.feature_rgb_file[filename]
        self.feature_flow=self.feature_flow_file[filename]
        duration = min(len(self.feature_rgb), len(self.feature_flow))
        
        for i in range(0,duration, frame_length):
            self.samples.append(i)
        
        self.label_seq = torch.zeros([duration, num_classes],dtype=torch.float32)
        self.label_seq[:,0]=1
        with open(label_dir, "r") as fp:
            for string in fp:
                string = string.split()
                if(string[0] != filename):
                    continue
                label = torch.zeros(num_classes, dtype=torch.float32)
                label[int(string[3])]=1
                
                st=int(int(string[1])/self.frame_interval)
                ed=int(int(string[2])/self.frame_interval)+1
                if(st < self.label_seq.size(0) and ed < self.label_seq.size(0)):
                    self.label_seq[st:ed,0]=0
                    self.label_seq[st:ed]+=label
                    self.label_seq=torch.clamp(self.label_seq, min=0, max=1)  
        
        
    def __getitem__(self, index):
        framedata = []
        idx = self.samples[index]
        
        duration = min(len(self.feature_rgb), len(self.feature_flow))
        feature = np.append(self.feature_rgb[:duration,:],self.feature_flow[:duration,:], axis=1)
        
        if(duration < idx+self.frame_length):
            start_frame=idx
            frames = feature[start_frame:start_frame+self.frame_length]
            frames_padding = np.zeros((idx+self.frame_length-duration,feature.shape[1]), dtype=np.float32)
            frames = np.append(frames,frames_padding, axis=0)
        else:
            start_frame = idx
            
            frames = feature[start_frame:start_frame+self.frame_length]
            
        assert(len(frames)==self.frame_length)
        return {'data': torch.from_numpy(np.array(frames)), 'idx': idx, 'label': self.label_seq[self.frame_length-1]}
        
    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return self.samples
