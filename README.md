# ACFNet: Attentional Class Feature Network for Semantic Segmentation

## This code is a unofficial implementation of the experiments on Cityscapes in the [ACFNet](https://arxiv.org/abs/1909.09408). 
 

## Introduction


## Architecture

    
### Requirements

Python 3.7

4 x 12g GPUs (e.g. TITAN XP)

```bash
# Install **Pytorch-1.1**
$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# Install **Apex**
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install **Inplace-ABN**
$ git clone https://github.com/mapillary/inplace_abn.git
$ cd inplace_abn
$ python setup.py install
```

### Dataset and pretrained model

Plesae download cityscapes dataset and unzip the dataset into `YOUR_CS_PATH`.

I use the ECA-Net101. Please download [ECA-Net101 pretrained](https://github.com/BangguWu/ECANet), and put it into `dataset` folder.

### Training and Evaluation
Training script.
```bash
python train.py --data-dir ${YOUR_CS_PATH} --random-mirror --random-scale --restore-from ./dataset/*.pth --gpu 0,1,2,3 --learning-rate 1e-2 --input-size 769,769 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 
``` 

【**Recommend**】You can also open the OHEM flag to reduce the performance gap between val and test set.
```bash
python train.py --data-dir ${YOUR_CS_PATH} --random-mirror --random-scale --restore-from ./dataset/*.pth --gpu 0,1,2,3 --learning-rate 1e-2 --input-size 769,769 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000
``` 

Evaluation script.
```bash
python evaluate.py --data-dir ${YOUR_CS_PATH} --restore-from snapshots/CS_scenes_60000.pth --gpu 0
``` 

All in one.
```bash
./run_local.sh YOUR_CS_PATH
``` 



## Thanks to the Third Party Libs
Self-attention related methods:
[Criss-Cross Network](https://github.com/speedinghzl/CCNet)
[Object Context Network](https://github.com/PkuRainBow/OCNet)    
[Dual Attention Network](https://github.com/junfu1115/DANet)   
Semantic segmentation toolboxs:   
[pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)   
[semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)   
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
