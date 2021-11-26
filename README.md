# Future Anticipation and Temporally Smoothing network (FATSnet)
**Temporally Smooth Online Action Detection using Cycle-consistent Future Anticipation**  
Young Hwi Kim, Seonghyeon Nam, Seon Joo Kim  
[[`arXiv`](https://arxiv.org/abs/2104.08030)]

## Updates
**25 Nov, 2021**: Initial update

## Installation

### Prerequisites
- Ubuntu 16.04  
- Python 2.7.17   
- CUDA 10.0  

### Requirements
- pytorch==1.4.0  
- numpy==1.16.6
- h5py==2.10.0

## Training

### Input Features
We provide the Kinetics pre-trained feature of THUMOS'14 dataset. The extracted features can be downloaded from [here](https://drive.google.com/file/d/1GwQjMq0Eyc3XWljeeaSqwbTal5y76Xwy/view?usp=sharing). Files should be located in 'data/'.  
The feature that is pre-trained on Activitynet can be downloaded from [here](https://drive.google.com/file/d/1trAN8g-xRyIIFMXEsqmgowD2CJcEGY10/view?usp=sharing).

### Trained Model
The trained models that used Kinetics pre-trained feature can be downloaded from [here](https://drive.google.com/file/d/1WeHyFq1v-Rch9qht_ACF2TLXGKizR_FI/view?usp=sharing). Files should be located in 'checkpoints/'. 
The Activitynet version can be downloaded from [here](https://drive.google.com/file/d/1wr-NXBW5LVNQ0WdkRmzUK7XPq_Y23VoV/view?usp=sharing).

### Training Model
For Kinetics pre-trained input feature,
```
python train.py --gen_feature_len=12
```
For Activitynet pre-trained input feature,
```
python train.py --gen_feature_len=8 --feature_size=3072
```


## Testing
For Kinetics pre-trainedd input feature,
```
python prediction.py
python eval_map.py
```
For Activitynet pre-trained input feature,
```
python prediction.py --feature_size=3072
python eval_map.py
```


| Dataset | Feature | mAP | 
|:--------------:|:--------------:|:--------------:| 
| THUMOS'14 | TwoStream-Anet | 51.6 |
| THUMOS'14 | TwoStream-Kinetics | 59.0 |


## Citing FATSnet
Please cite our paper in your publications if it helps your research:

```BibTeX
@article{kim2021temporally,
  title={Temporally smooth online action detection using cycle-consistent future anticipation},
  author={Kim, Young Hwi and Nam, Seonghyeon and Kim, Seon Joo},
  journal={Pattern Recognition},
  volume={116},
  pages={107954},
  year={2021},
  publisher={Elsevier}
}
```
