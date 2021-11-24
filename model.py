import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, feature_len, label_num):
        super(model, self).__init__()      
        self.forward_cell = nn.GRUCell(feature_len, 4096 , bias=True)
        self.backward_cell = nn.GRUCell(feature_len, 4096 , bias=True)
        self.action_cell = nn.GRUCell(feature_len,4096, bias=True)
        
        self.fc_feature = nn.Linear(4096,feature_len, bias=True)
        self.fc_action1 = nn.Linear(4096,feature_len, bias=True)
        self.fc_action2 = nn.Linear(feature_len,label_num, bias=True)
        
        self.attention1 = nn.Linear(feature_len, int(feature_len/4), bias=False)
        self.attention2 = nn.Linear(int(feature_len/4), 2, bias=False)
        
        self.avgpool = nn.AvgPool3d((1,7,7), stride=1)
        self.maxpool = nn.MaxPool3d((3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.finalavg = nn.AdaptiveAvgPool2d((1,label_num))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)        
    
    def decoder_forward(self, x,h):
        h =  self.forward_cell(x,h)
        gen_feature = self.fc_feature(h)
        return gen_feature, h
        
    def decoder_backward(self, x,h):
        h =  self.backward_cell(x,h)
        gen_feature = self.fc_feature(h)
        return gen_feature, h
                
    def forward(self, x, prev_prob, gen_feature_len):
        input_size = x.size()
        
        with torch.no_grad():
            h=torch.zeros(x.size(0), 4096, requires_grad = True).cuda()
            for i in range(0, x.size(1)):
                gen_feature, h = self.decoder_forward(x[:,i,:], h)
                
            syn_feature = [gen_feature]
            for i in range(0, gen_feature_len-1):
                gen_feature, h = self.decoder_forward(gen_feature,h)
                syn_feature.append(gen_feature)
                
        actions = []
        
        feature = x
        h=torch.zeros(input_size[0], 4096, requires_grad = True).cuda()
        for i in range(0, input_size[1]):
            h = self.action_cell(feature[:,i,:], h)
        
        for i in range(0, gen_feature_len):
            h = self.action_cell(syn_feature[i], h)
        
        feature = self.relu( self.fc_action1(h))
        cur_prob = self.fc_action2(feature)
        
        if(prev_prob.ne(0).any()):
            last_x = x[:,-1,:]
            old_new_weight = self.relu(self.attention1(last_x))
            old_new_weight = self.softmax(self.attention2(old_new_weight))
            old_new_weight = old_new_weight.view(old_new_weight.size(0),1,old_new_weight.size(1))
            
            merge_prob = torch.stack([prev_prob, cur_prob], dim=1)
            action_prob = torch.bmm(old_new_weight,merge_prob).squeeze(1)
            old_new_weight = old_new_weight.squeeze(1)
            
            return action_prob
        else:
            dummy_weight=torch.zeros(input_size[0], 2).cuda()
            dummy_weight[:,1]=1
            return cur_prob
        
    def forward_unroll(self, feature, h_dec, h_act, prev_prob, gen_feature_len):
        with torch.no_grad():
            gen_feature, h_dec = self.decoder_forward(feature, h_dec)
                
            syn_feature = [gen_feature]
            h = h_dec.clone()
            for i in range(0, gen_feature_len-1):
                gen_feature, h = self.decoder_forward(gen_feature,h)
                syn_feature.append(gen_feature)
                                        
        h_act = self.action_cell(feature, h_act)
            
        h = h_act.clone()
        for i in range(0, gen_feature_len):
            h = self.action_cell(syn_feature[i], h)
        
        feature_action = self.relu( self.fc_action1(h))
        cur_prob = self.fc_action2(feature_action)
        
        if(prev_prob.ne(0).any()):
            old_new_weight = self.relu(self.attention1(feature))
            old_new_weight = self.softmax(self.attention2(old_new_weight))
            old_new_weight = old_new_weight.view(old_new_weight.size(0),1,old_new_weight.size(1))
            
            merge_prob = torch.stack([prev_prob, cur_prob], dim=1)
            action_prob = torch.bmm(old_new_weight,merge_prob).squeeze(1)
            old_new_weight = old_new_weight.squeeze(1)
            
            return action_prob, h_dec, h_act, old_new_weight
        else:
            dummy_weight=torch.zeros(feature.size(0), 2).cuda()
            dummy_weight[:,1]=1
            return cur_prob, h_dec, h_act, dummy_weight
        
