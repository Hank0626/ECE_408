# ECE408/CS483 Final Project: Optimizing Convolutional Layers with CUDA
## Introduction
Welcome to the Fall 2022 ECE408 / CS483 course project! In this project, you will be implementing and optimizing the forward-pass of a convolutional layer using CUDA. The primary goal is to enhance your understanding of CUDA and optimization techniques while gaining practical experience with profiling tools like Nsight Systems (nsys) and Nsight-Compute (nv-nsight-cu).

You will work individually on this project, which revolves around a modified version of the LeNet-5 architecture used for convolutional neural networks (CNNs). CNNs are widely used in various machine learning tasks, such as image classification, object detection, natural language processing, and recommendation systems.

The project will utilize the Fashion MNIST dataset and the mini-dnn-cpp (Mini-DNN) framework for implementing the modified LeNet-5. You will be optimizing the CUDA implementation of the convolutional layer for layers C1 and C3 in the LeNet-5 architecture.

## Milestones
The project is divided into several milestones, with each milestone building upon the previous one. For each milestone, you will need to submit a report on Canvas, detailing your progress and achievements.

The first milestone focuses on Rai installation, CPU convolution, and profiling. You will create a CPU convolution implementation, profile it using gprof, and run the Mini-DNN forward pass on the CPU.

For detailed instructions, please refer to the original [readme file](./Project/README.md).

## Learning Objectives
By the end of this project, you will have demonstrated:
* Command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolutional layer forward pass.
* Practical experience in analyzing and fine-tuning CUDA kernels using profiling tools like Nsight Systems (nsys) and Nsight-Compute (nv-nsight-cu).

## Collaboration and Academic Integrity
Please remember that you are expected to adhere to the University of Illinois academic integrity standards. Do not attempt to subvert any performance-measurement aspects of the final project. If you are unsure about whether something meets those guidelines, consult a member of the teaching staff.