# Traffic Sign Classification

Classify traffic signs using traditional machine learning method and deep learning methods. [course project of "Media and Recognition" of EE, Tsinghua University]

**See the [report](report/report.pdf)**.

## Introduction

This cource project includes 4 tasks. Task 1 requires us to classify traffic signs using traditional machine learning method, task 2 requires us to classify using deep learning method, task 3 requires us to perform single example classification and task 4 requires us to detect traffic signs and then classify.

I am responsible for task 1 & 2, so this repository only consists of code and report of these 2 tasks. If my teammates decide to public the remaining tasks on GitHub, I will then add the links.

## Dataset

You can download the dataset from [Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/b621dc639e7d4be6ba50/) or [Google Drive](https://drive.google.com/drive/folders/1vQubmgHQuVFoZ7JCpLwyG3PM16rHhjYU).

## Requirement

Task 1 requires

- numpy
- cv2
- tqdm
- scipy
- sklearn

Task 2 requires

- torch
- pytorch_lightning
- torchvision
- PIL
- tqdm

## Results

Our work has 95.16% accuracy of task 1, and 97.89% accuracy of task 2. Although we have relatively high accuracy of task 1, the accuracy of task 2 is not that high enough. The reason is that I adopted the network structure in this paper [[content](https://www.sciencedirect.com/science/article/abs/pii/S0893608018300054?via%3Dihub), [code (Lua)](https://github.com/aarcosg/tsr-torch)], but did not have time to shrink its size to fit our dataset (Our dataset is much smaller so this will apparently cause overfitting problems).
