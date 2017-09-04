# Hackathon Castelló
Aquest repositori es per a la participació en el [Hackathon de Castelló 2017](http://www.hackathoncastellon.com), en el repte d'Inteligencia Artificial - Machine Learning.

## Installation

### Installing virtualenv

Take the following steps to install Pytorch and TensorFlow with Virtualenv:

1. Start a terminal (a shell). You'll perform all subsequent steps in this shell.

2. Install pip and virtualenv by issuing the following commands:

```
 $ sudo easy_install pip
 $ sudo pip install --upgrade virtualenv 
```

Create a virtualenv environment by issuing a command of one of the following formats:

```
 $ virtualenv --system-site-packages -p python3 venv
```

where targetDirectory identifies the top of the virtualenv tree. Our instructions assume that targetDirectory is ~/Documents/Development/hackathon-castellon/venv, but you may choose any directory.

Activate the virtualenv environment by issuing one of the following commands:

```
$ source ~/Documents/Development/hackathon-castellon/venv/bin/activate      # If using bash, sh, ksh, or zsh
```

The preceding source command should change your prompt to the following:

```
 (venv)$ 
```

Ensure pip ≥8.1 is installed:

```
 (venv)$ easy_install -U pip
```

### Installing TensorFlow + Keras

Issue one of the following commands to install TensorFlow and all the packages that TensorFlow requires into the active Virtualenv environment:

```
 (venv)$ pip3 install --upgrade tensorflow
 (venv)$ pip3 install --upgrade pip install tensorboard_logger
 (venv)$ pip install keras
```

### Installing Pytorch

```
 (venv)$ pip install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl 
 (venv)$ pip install torchvision 
```

### Installing auxiliar libraries
```
 (venv)$ pip install python-mnist
 (venv)$ pip install scikit-learn
 (venv)$ pip install psutil
```

## Organització d'algoritmes

En aquest repositori tenim models de classificació d'imatges de tres llibreries: Pytorch, TensorFlow i Keras. En cada lliberia tenim els següents algorismes:

* Pytorch
1. SimpleNet
2. [WRN - Wide Residual Networks](https://arxiv.org/pdf/1605.07146)
3. [ResNet - Residual Networks](https://deepmlblog.wordpress.com/2016/01/05/residual-networks-in-torch-mnist/), [Github](https://github.com/KaimingHe/deep-residual-networks)
4. [DenseNet - Densely Connected Convolutional Networks](https://github.com/liuzhuang13/DenseNet)

* Keras
1. [VGG - Very Deep Convolutional Network](https://arxiv.org/abs/1409.1556)

* Tensor Flow
1. [Comparativa entre ResNets, HighwayNets i DenseNets](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32)

## Running

Per a executar els algoritmes, entra en les carpetes corresponents i executar:

```python
python train.py
```

## Notes
https://github.com/zalandoresearch/fashion-mnist      
https://github.com/kuangliu/pytorch-cifar

### Models Pytorch
[DenseNet and WRN](https://github.com/ajbrock/FreezeOut)
[ResNet](https://github.com/kefth/fashion-mnist)
[Dual Path Networks](https://github.com/Queequeg92/DualPathNet)

### Models TensorFlow
https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
https://github.com/hwalsuklee/tensorflow-generative-model-collections

### Models Keras
https://github.com/QuantumLiu/fashion-mnist-demo-by-Keras
https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb