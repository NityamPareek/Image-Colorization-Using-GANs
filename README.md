# Image-Colorization-Using-GANs
# Description
This repository contains notebooks on training Generative Adversarial Networks (GANs) to tackle the task of image colorization.

# Introduction
Image-to-Image Translation with Conditional Adversarial Networks paper, which is known by the name pix2pix, proposed a general solution to many image-to-image tasks in deep learning, in which one of those was colorization. In this approach two losses are used: L1 loss, which makes it a regression task, and an adversarial (GAN) loss, which helps to solve the problem in an unsupervised manner by assigning the outputs a number indicating how "real" they look.

We use 2 different approaches for this task. 
1. Implementing the paper as it is, and training both the generator and discriminator from scratch together.
2. Pretraining the generator separately in a supervised and deterministic manner to avoid the problem of "the blind leading the blind" in the GAN game where neither generator nor discriminator knows anything about the task at the beginning of training. I'm going to use a pretrained ResNet18 as the backbone of my U-Net and to accomplish the second stage of pretraining, I am going to train the U-Net on the training set with only L1 Loss. Then I will move to the combined adversarial and L1 loss, as done previously.

# Theory

## 1. Generative adversarial networks (GAN)

A generative adversarial network (GAN) is a type of deep learning network that can generate data with similar characteristics as the input training data.

A GAN consists of two networks that train together:

* Generator — Given a vector of random values as input, this network generates data with the same structure as the training data.

* Discriminator — Given batches of data containing observations from both the training data, and generated data from the generator, this network attempts to classify the observations as "real" or "generated".

<center><img src="https://it.mathworks.com/help/examples/nnet/win64/TrainConditionalGenerativeAdversarialNetworkCGANExample_01.png"/> </center>

## 2. Conditional Generative adversarial networks (cGAN)

A conditional generative adversarial network (CGAN) is a type of GAN that also takes advantage of labels during the training process.

* Generator — Given a label and random array as input, this network generates data with the same structure as the training data observations corresponding to the same label.

* Discriminator — Given batches of labeled data containing observations from both the training data and generated data from the generator, this network attempts to classify the observations as "real" or "generated".

<center><img src="https://it.mathworks.com/help/examples/nnet/win64/TrainConditionalGenerativeAdversarialNetworkCGANExample_02.png"/> </center>

To train a conditional GAN, train both networks simultaneously to maximize the performance of both:

* Train the generator to generate data that "fools" the discriminator.

* Train the discriminator to distinguish between real and generated data.

To maximize the performance of the generator, maximize the loss of the discriminator when given generated labeled data. That is, the objective of the generator is to generate labeled data that the discriminator classifies as "real".

To maximize the performance of the discriminator, minimize the loss of the discriminator when given batches of both real and generated labeled data. That is, the objective of the discriminator is to not be "fooled" by the generator.

Ideally, these strategies result in a generator that generates convincingly realistic data that corresponds to the input labels and a discriminator that has learned strong feature representations that are characteristic of the training data for each label.

source : https://it.mathworks.com/help/deeplearning/ug/train-conditional-generative-adversarial-network.html

## 3. Why choosing cGAN over GAN

Conditional Generative Adversarial Networks (CGANs) are an extension of standard Generative Adversarial Networks (GANs) that are designed to handle conditional data. A CGAN consists of a generator network and a discriminator network, just like a standard GAN. However, in a CGAN, the generator and discriminator are both conditioned on some additional input data. This additional input data can be used to control the output of the generator, allowing it to produce more specific or customized results.

There are several reasons why a CGAN can be better than a standard GAN:

1. Control over the generated data: In a CGAN, the generator's output is conditioned on the input data, which allows the model to be more specific and controlled in its output. For example, if the input is a grayscale image, the model can colorize it to a specific color scheme.

2. Improved stability and training: Because the generator is conditioned on additional input data, it can be easier to train and more stable than a standard GAN. This is because the generator is able to focus on a specific subset of the data, rather than trying to generate all possible outputs.

3. Handling missing data: CGANs are well suited for handling missing data or data with missing modalities. The additional input data can be used to condition the generator to produce plausible outputs for the missing data.

4. Handling multiple classes: CGANs can be used to generate data for multiple classes in a one-to-many mapping, where the generator is conditioned on the class label and produces an image from that class.

5. Handling conditional data: In some tasks, the data is conditional, such as in image-to-image translation, where the output is conditioned on the input. CGANs can handle this kind of conditional data very well.

It's important to note that in some tasks a GAN might be enough or even better than a CGAN, it depends on the task and the data.

# Architecture as Proposed by Paper

## 1. Generator

<center> <img src="https://i.imgur.com/k6ErEni.png"></center>

## 2. Discriminator

<center> <img src="https://i.imgur.com/rG6DjQA.png"></center>

# Loss

## Loss Function to be optimized

![image](https://github.com/NityamPareek/Pix2Pix-Image-Colorization-Using-GANs/assets/97893479/bf0bd0fd-974b-4c75-a8f8-aac674325d24)
 
## L1 loss

![image](https://github.com/NityamPareek/Pix2Pix-Image-Colorization-Using-GANs/assets/97893479/b9f2f997-4008-4d39-93ed-dc27eb4e84bb)
 
## GAN Loss

![image](https://github.com/NityamPareek/Pix2Pix-Image-Colorization-Using-GANs/assets/97893479/98073a7f-fe16-43d5-8cfa-78cd9436ea54)
 
x -> grayscale image (the condition introduced)

y -> 2 channel output of generator

z -> input noise of generator

G -> Generator Model

D -> Discriminator Model

# Improving the Results

In order to avoid a case of "blind leading the blind", I took the following measures:
1. Replaced the backbone of the U-Net with a ResNet18 pre-trained on the ImageNet dataset. 
2. Pre-Trained the generator on the entire dataset for 20 epochs with L1 loss
3. Trained the entire GAN for 20 epochs

# Models
Download my trained models from: https://drive.google.com/drive/folders/15c52V5yaaZILAzhxL3e6j5wz3bDisOQl?usp=sharing

# Results
![image](https://github.com/NityamPareek/Image-Colorization-Using-GANs/assets/97893479/4a35a58e-452a-47b6-a2e7-515c306f5441)

# Conclusion and Final Thoughts
We can see that the GAN is performing well on the data even with a small training dataset and less number of iterations.



