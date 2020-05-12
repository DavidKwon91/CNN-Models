# Convolutional Neural Network

This repository is to introduce some classic CNN models. Also, their variants or customized version of them are discussed. 

Most of their architecture are structured for the shape of various image datasets, such as ImageNet or cifar. 
Their architectures are customizable with these modules in this repository.

You will be able to build original or customized version of the classic CNN models, train, evaluate, and predict them with the modules. 

They are developed with Keras. 

# Models

1. Lenet
* [Lenet Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

* You can customize..
    - activation
    - weight initializer
    - regularizer
    - optimizer
    - structure (depends on the input shape)
    
* Original Architecture

|Layer           |Maps   |Size   |Kernel Size|stride |Activation|
|----------------|:-----:|:-----:|:---------:|:-----:|:--------:|
|Input           |1      |32 x 32|           |       |          |
|Conv1           |6      |28 x 28|5 x 5      |1      |tanh      |
|Avg Pooling2D   |6      |14 x 14|2 x 2      |2      |          |
|Conv2           |16     |10 x 10|5 x 5      |1      |tanh      |
|Avg Pooling2D   |16     |5 x 5  |2 x 2      |2      |          |
|Conv2           |120    |1 x 1  |5 x 5      |1      |tanh      |
|FC              |       |84     |           |       |tanh      |
|FC              |       |10     |           |       |softmax   |


2. Alexnet
* [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

The input shape that is stated in the paper is 224 x 224 x 3, but it must be 227 x 227 x 3 as stated in [cs231n](https://cs231n.github.io/convolutional-networks/)

* Two versions of Alexnet available
    1. orginial version
    2. mini version


* Original Architecture

|Layer          |Maps |Size     |Kernel Size|Strides|Activation|
|:-------------:|:---:|:-------:|:---------:|:-----:|:--------:|
|Input          |3    |227 x 227|           |       |          |
|Conv2(BN, Act) |96   |55 x 55  |11 x 11    |4      |relu      |
|MaxPooling     |96   |27 x 27  |3 x 3      |2      |relu      |
|Conv2(BN, Act) |256  |27 x 27  |5 x 5      |1      |relu      |
|MaxPooling     |256  |13 x 13  |3 x 3      |2      |relu      |
|Conv2(Act)     |384  |13 x 13  |3 x 3      |1      |relu      |
|Conv2(Act)     |384  |13 x 13  |3 x 3      |1      |relu      |
|Conv2(Act)     |256  |13 x 13  |3 x 3      |1      |relu      |
|MaxPooling     |256  |6 x 6    |3 x 3      |2      |relu      |
|FC(Dropout)    |     |4096     |           |       |relu      |
|FC(Dropout)    |     |4096     |           |       |relu      |
|FC             |     |1000     |           |       |softmax   |


- Batch Normalization is used instead of LRN introduced in the paper
- Normalization is applied into the first and the second conv layer as stated in the paper


3. 