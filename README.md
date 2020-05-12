# Convolutional Neural Network

This repository is to show the algorithms of some classic CNN models. Most of their architecture are structured for the shape of various image datasets, such as ImageNet or cifar. 
Their architectures are customizable in these modules of this repository, even though some models are not available to be customized, but have different versions of them. 

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

|Layer           |Maps   |Size   |Kernel Size|
|----------------|:-----:|:-----:|:---------:|
|input           |1      |32 x 32|           |
|Conv1           |6      |28 x 28|5 x 5      |
|Avg Pooling2D   |6      |14 x 14|2 x 2      |
|Conv2           |16     |10 x 10|5 x 5      |
|Avg Pooling2D   |16     |5 x 5  |2 x 2      |
|Conv2           |120    |1 x 1  |5 x 5      |
|Flatten         |120    |120    |           |
|Dense           |84     |84     |           |
|Dense           |10     |10     |           |


2. Alexnet
* [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

The input shape that is stated in the paper is 224 x 224 x 3, but it must be 227 x 227 x 3 as stated in [cs231n](https://cs231n.github.io/convolutional-networks/)

* Two versions of Alexnet available
    1. orginial version
    2. mini version
        

3. 