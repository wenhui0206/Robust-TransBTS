# Semi-supervised Learning using Robust Loss

This is the implementation of the paper:  [Semi-supervised Learning using Robust Loss](https://arxiv.org/abs/2203.01524)

This repo is forked from the official implementation for [TransBTS: Multimodal Brain Tumor Segmentation Using Transformer](https://arxiv.org/pdf/2103.04430.pdf). The multimodal brain tumor datasets (BraTS 2019 & BraTS 2020) could be acquired from [here](https://ipp.cbica.upenn.edu/).

## Python Scripts
- `python -m torch.distributed.launch --nproc_per_node=2 train_cv.py --corrupt_r=0.0 --train_partial=False --beta=0.0 --experiment='Upperbound_lw' --fold=0`
- `python /scratch1/wenhuicu/robust_seg/TransBTS/validation.py --test_file='model_epoch_last.pth' --valid_file='test_list.txt' --submission='' --experiment='Upperbound_lw.5_f0' --csv_name='Upperbound_lw_f0.csv' `
  
### TransBTS
![TransBTS](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/TransBTS.PNG "TransBTS")
Architecture of 3D TransBTS.

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel

## Data preprocess
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

`python3 preprocess.py`

## Training
Run the training script on BraTS dataset. Distributed training is available for training the proposed TransBTS, where --nproc_per_node decides the numer of gpus and --master_port implys the port number. 

- train_cv.py is the training file mainly used for baseline training, CE loss, and Robust loss training, and performs 3-fold cross validation. To train on different folds, specify the fold number. 

- train_main.py is same as train_cv.py except that it does not have cross-validation part.

- train_cps.py implements [cross pseudo supervision](https://github.com/charlesCXK/TorchSemiSeg/tree/f67b37362ad019570fe48c5884187ea85f2cc045). For comparison with robust loss, it is based on train_cv.py

- train_2losses.py tried to use ce loss for gt labels, robust loss for pseudo labels. But the results were not improving.


`python -m torch.distributed.launch --nproc_per_node=2 train_cv.py`

## Testing 

Run python validation.py

- validation.py is the one used for performance evaluation. It calculates Dice Scores and Hausdorff Distance. Results are saved in a csv file, you can use calc_mean_var() function in plot.py file to calculate mean dices across 3 folds.

- predict.py has the code for actual model evaluation and metric calculation. We use validate_performance() function to calculate dices and hd, and save the mean in a csv file separately for each fold. compare_performance() is used to generate predicted segmentation maps and save them if specify --submission argument. Also, compare_performance saves all dice scores of each subject in a txt file for later analysis.


## Model Architecture
- TransBTS_downsample8x_skipconnection_lw.py is the one uses one layer in the transformer module, and half-sized hidden layer (last flatten layer)



## Reference
1.[setr-pytorch](https://github.com/gupta-abhay/setr-pytorch)

2.[BraTS2017](https://github.com/MIC-DKFZ/BraTS2017)


