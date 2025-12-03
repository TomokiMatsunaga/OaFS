# Multitrack Music Transcription Based on Joint Learning of Onset and Frame Streams
OaFS (Onset and Frame Streams) is a multitrack music transcription framework that integrates a residual U-Net (ResUnet) with a hierarchical Perceiver.

## Usage
#### Install
  ```
  git clone https://github.com/TomokiMatsunaga/OaFS.git
  ```
  ```
  cd OaFS
  ```
  ```
  conda env create -f environment.yml
  ```
  ```
  conda activate OaFS
  ```
- Install `PyTorch>=1.12` following the [official installation instructions](https://pytorch.org/get-started/locally/)
- Install `TensorFlow==2.12.0` following the [official installation instructions](https://www.tensorflow.org/install/pip)


#### Create Feature Files
  ```
  python main.py --dataset 'dataset name' --file_path 'path name'
  ```
#### Train
  ```
  python learning.py --train_mode 0 --train_feature_path 'feature path for train' --val_feature_path 'feature path for validation'
  ```
#### Prediction
  ```
  python learning.py --train_mode 1 --evaluation_mode 0 --dataset_list 'dataset list'
  ```
#### Evaluation
  ```
  python MLCFP.py --train_mode 1 --evaluation_mode 1 --dataset_list 'dataset list'
  ```

The trained model used in the experiment can be found here: https://huggingface.co/tmatsu11/OaFS


## Citation
  Under review
