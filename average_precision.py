import numpy as np

conf_threshold =0.0


def ap(label, predicted):
    frame = np.stack([label,predicted], axis=1)
    
    num_frame = label.shape[0]
    num_label = label.shape[1]
    
    APs = []
    for label_idx in range(num_label):
        target_frame = frame[:,:,label_idx]
        
        target_frame = target_frame[target_frame[:,1].argsort()][::-1]
        
        sum_prec=0
        total_positive=target_frame[:,0].sum()
        
        tp=0.0
        fp=0.0
        for k in range(0, num_frame):
            if(target_frame[k,1]>conf_threshold):
                if(target_frame[k,0]==0):
                    fp+=1
                if(target_frame[k,0]==1):
                    tp+=1
                    sum_prec += tp/(tp+fp)
                
        APs.append(sum_prec/total_positive)
        
    return APs
    
def calibrated_ap(label, predicted):
    frame = np.stack([label,predicted], axis=1)
    
    num_frame = label.shape[0]
    num_label = label.shape[1]
    
    APs = []
    
    num_negative = sum(label[:,0])
    num_positive = num_frame-num_negative
    w = num_negative/num_positive
    
    for label_idx in range(num_label):
        target_frame = frame[:,:,label_idx]
        
        target_frame = target_frame[target_frame[:,1].argsort()][::-1]
        
        sum_prec=0
        total_positive=target_frame[:,0].sum()
        num_positive =total_positive
        num_negative = num_frame-num_positive
        w=num_negative/num_positive
        
        tp=0.0
        fp=0.0
        for k in range(0, num_frame):
            if(target_frame[k,0]==1):
                num_positive+=1
            if(target_frame[k,1]>conf_threshold):
                if(target_frame[k,0]==0):
                    fp+=1
                if(target_frame[k,0]==1):
                    tp+=1
                    sum_prec += w*tp/(w*tp+fp)
                
        APs.append(sum_prec/total_positive)
        
    return APs
