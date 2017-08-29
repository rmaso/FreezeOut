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
 $ virtualenv --system-site-packages targetDirectory # for Python 2.7
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

### Installing TensorFlow

Issue one of the following commands to install TensorFlow and all the packages that TensorFlow requires into the active Virtualenv environment:

```
 (venv)$ pip install --upgrade tensorflow      # for Python 2.7
```

### Installing Pytorch

```
 (venv)$ pip install http://download.pytorch.org/whl/torch-0.2.0.post3-cp27-none-macosx_10_7_x86_64.whl 
 (venv)$ pip install torchvision 
```

## Running

To run with default parameters, simply call

```python
python train.py
```

This will by default download CIFAR-100, split it into train, valid, and test sets, then train a k=12 L=76 DenseNet-BC using SGD with Nesterov Momentum.

## Notes
https://github.com/zalandoresearch/fashion-mnist      
https://github.com/kuangliu/pytorch-cifar
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

https://github.com/hwalsuklee/tensorflow-generative-model-collections

