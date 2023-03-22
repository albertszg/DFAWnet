# DFAWnet

## a nueral network constructed under the frame work of signal processing informed neural network (SPINN)
## 故障感知去噪小波网络: 基于信号处理启发的神经网络框架
## code for DFAWnet  
### 1. parameter setting: main.py
### 2.training program:utils/train_utils.py
### 3.Spectral kurtosis based Loss: physics_loss/kurtosis.py
### 4.Energy-based channel (scale) selection layer (attention weighting layer):paper_model/weighting_layer.py 
### 5.The whole DFAWnet:paper_model/DFWnet.py
### 6.Fused 1/2/3 wavelet convolution:fusion_modules/fuse_conv_wavelet.py 
### 7.Dynamic hard thresholding mask:dynconv.py


**if you are intersted in this work, please cite**

@article{shang2023denoising,
  title={Denoising Fault-Aware Wavelet Network: A Signal Processing Informed Neural Network for Fault Diagnosis},
  author={Shang, Zuogang and Zhao, Zhibin and Yan, Ruqiang},
  journal={Chinese Journal of Mechanical Engineering},
  volume={36},
  number={1},
  pages={9},
  year={2023},
  publisher={Springer}
}




