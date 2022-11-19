# NeurIPS 2022 submission: AdaSR: Adaptive Block Quantization for Enhancing Super-resolution on Compressed Images.

This is an official implementation of the paper AdaSR: Adaptive Block Quantization for Enhancing Super-resolution on Compressed Images.

This repository is used to reproduce the results of paper and includes two kinds of categories: 1) Evaluate the AdaSR, 2) Train the QP-NET.

## Prerequisites

* OS: Ubuntu 16.04 or higher versions
* HW: NVIDIA GPU
* Virtual Environment for Python: Anaconda3

## Setup
* Setup the python virtual enviroment
```
conda env create --file environment.yml
conda activate AdaSR
```
* Setup the python environment variable
```
export PYTHONPATH="${PYTHONPATH}:{Where AdaSR is stored}/AdaSR-public"
```

## Evaluate the AdaSR

### 1. Download the testing dataset and statistics
Download the [test4K](https://drive.google.com/file/d/1xOoPARGRvXTtn4a-QLBcxMiUn8TLKfgV/view?usp=sharing) dataset and [size&quality statistics](https://drive.google.com/file/d/1jPBk4lzlYPgFQaEHTdLq3HgnlKdOiMyj/view?usp=sharing) from Google Drive and move the directory to dataset directory.

```
./dataset
├── test4k
├── profile
  ├── size_qual_stats.csv
```

(The test4K dataset is sampled from [DIV8K](https://competitions.codalab.org/competitions/22217#participate)) (index 1301-1400). This dataset is also used in [ClassSR](https://github.com/XPixelGroup/ClassSR#readme) for testing.)

### 2. Download the pretrained SR models and QP-NET
Download the [pretrained SR models](https://drive.google.com/drive/folders/1jXteaNeWl0ZSyzB756WZeinXSY8kIDXc?usp=sharing) from Google Drive and move the pretrained models to pretrained directory.

```
./pretrained
├── FSRCNN_B4_F16_S4.pth
├── FSRCNN_B4_F36_S4.pth
├── ...
├── EDSR_B8_F48_S4.pth
├── ...
```

Download the [pretrained QP-NET](https://drive.google.com/drive/folders/1tK9UGbfnpFOhWSRlYxC0sszj1NznOixO?usp=sharing) from Google Drive and move the pretrained models to checkpoint directory.

```
./checkpoint
├── QPNET_L2_C128_32.pt
```

### 3. Run evaluating for AdaSR
```
./script/eval.sh
```

If you want to run the evaluation more sophisticated, you can use the code below.

```
python ./eval/test_AdaSR.py --quality {$JPEG QP} --device {$GPU Device Number} --sr_model_name {$SR model name (e.g. edsr, fsrcnn, carn)} --use_cpu (If you want to use CPU instead of GPU) --save_image (If you want to save the result images)
```

### 4. Results
```
./result
├── test4k
  ├── ${sr_model_name}
    ├── ${scale}
      ├── ${quality}
        ├── log
          ├── log.txt (Computing and PSNR results of AdaSR and Baseline)
        ├── visual (SR results of AdaSR and Baseline)
```

## Train the QP-NET

### 1. Download the train and validation dataset
Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset for train and validation.

```
./dataset
├── DIV2K
  ├── DIV2K_train_HR
  ├── DIV2K_train_LR_bicubic
  ├── DIV2K_valid_HR
  ├── DIV2K_valid_LR_bicubic   
```

### 2. Generate patches and infos for training 

```
./script/prepare_training.sh
```

(Download pretrained SR models for generating training dataset)


### 3. Run the training code for QP-NET

```
./script/train_QP-NET.sh
```

If you want to run the training more sophisticated, you can use the code below.

```
python ./src/dnn/train_qpnet.py --learning_rate {$learning rate} --batch_size {$mini batch size} --num_epochs {$number of epochs} --num_channels {$number of channels of QP-NET} --num_layers {$number of layers of QP-NET} --device {$GPU Device Number} --use_cpu (If you want to use CPU instead of GPU) --save_image (If you want to save the result images)
```

### 4. Results
```
./checkpoint
├── (e.g.) QPNET_L2_C128_32.pt
```
