# Recorded experiment environment

## Thesis-ready sentence

All experiments were conducted on a Windows 11 workstation with an NVIDIA
GeForce MX450 GPU, using Python 3.7 and PyTorch 1.10.2 for OSNet training and
evaluation, and Python 3.14, PyTorch 2.13, and Transformers 5.15.1 for
training-free SigLIP2 inference.

## OSNet / Torchreid

- Repository: KaiyangZhou/deep-person-reid
- Repository commit: `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`
- Operating system: Microsoft Windows 11 Home China, 64-bit
- GPU: NVIDIA GeForce MX450, 2 GB VRAM
- Python: 3.7.12
- PyTorch: 1.10.2+cu113
- Torchvision: 0.11.3+cu113
- cuDNN: 8.2
- NumPy: 1.21.6
- SciPy: 1.7.3
- Scikit-learn: 1.0.2

The recorded Market-1501 training run used OSNet_x1_0 with ImageNet
pretraining, 256x128 inputs, random flip and random erase, label-smoothed
softmax loss, AMSGrad, learning rate 0.0015, cosine scheduling, batch size 8,
250 epochs, and random seed 1.

## SigLIP2

- Model: `google/siglip2-base-patch16-224`
- Python: 3.14.3
- Transformers: 5.15.1
- CPU environment: PyTorch 2.13.0+cpu
- GPU environment: PyTorch 2.13.0+cu130, CUDA available
- NumPy: 2.5.2
- Pillow: 12.3.0

The same pretrained semantic model and cached predictions were reused across
all downstream ablations; neither OSNet nor SigLIP2 was retrained.
