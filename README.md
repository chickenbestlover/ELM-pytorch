# ELM-pytorch
Extreme Learning Machine (ELM) implemented in Pytorch.

It's MNIST tutorial with basic ELM algorithm, Online Sequential ELM (OS-ELM), and Convolutional ELM.

You can run the code using cpu or gpu mode.

## Requirements
* Python 3.5+
* Pytorch 0.3.1+

## Extreme Learning Machine

__Usage:__
```
cd mnist

GPU mode: python main_ELM.py

CPU mode: python main_ELM.py --no-cuda
```
The training was completed in 2.0sec and the accuracy reached 97.77%.
(Geforce GTX1080Ti 11GB, #hidden neurons=7000)

In CPU mode, the training was completed in 26.92sec and the accuracy was the same.
(intel Core i7-6700K CPU 4.00GHz x 8 64GB RAM, #hidden neurons=7000)

If you do not have enough memory for the training process, reduce the number of hidden neurons and try again.

## Online Sequential Extreme Learning Machine

__Usage:__
```
cd mnist

GPU mode: python main_ELM.py

CPU mode: python main_ELM.py --no-cuda
```
The training was completed in 10.0sec and the accuracy reached 97.77%.
(Geforce GTX1080Ti 11GB, #hidden neurons=7000, batch_size=1000)

In CPU mode, the training was completed in 100.92sec and the accuracy was the same.
(intel Core i7-6700K CPU 4.00GHz x 8 64GB RAM, #hidden neurons=7000, batch_size=1000)

If you do not have enough memory for the training process, reduce the number of hidden neurons and try again.

## Convolutional Extreme Learning Machine

__Usage:__
```
cd mnist

GPU mode: python main_CNNELM.py

CPU mode: python main_CNNELM.py --no-cuda
```
The training was completed in 7.2sec and the accuracy reached 98.01%.
(Geforce GTX1080Ti 11GB, the code used almost all RAM.)

Network configuration

ConvLayer1: kernel_size=5, #channel=10, padding=1
PoolLayer1: kernel_size=2
ReluLayer1:
ConvLayer2: kernel_size=4, #channel=80, padding=1
PoolLayer2: kernel_size=2
ReluLayer2:
FCLayer:

In CPU mode, the training was completed in 177.92sec and the accuracy was 98.80%.
(intel Core i7-6700K CPU 4.00GHz x 8 64GB RAM, the code used almost all RAM.)

Network configuration

ConvLayer1: kernel_size=5, #channel=10, padding=1
PoolLayer1: kernel_size=2
ReluLayer1:
ConvLayer2: kernel_size=4, #channel=450, padding=1
PoolLayer2: kernel_size=2
ReluLayer2:
FCLayer:

If you do not have enough memory for the training process, reduce the number of hidden neurons and try again.


