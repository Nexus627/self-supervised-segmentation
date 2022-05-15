# Biological Image Segmentation Using Self Supervised Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Self-supervised Learning](#self-supervised-learning)
4. [Segmentation](#segmentation)
5. [Dataset](#dataset)
6. [Workflow](#workflow)
7. [Results](#results)
    1. [Experiment 1](#experiment-1)
    2. [Experiment 2](#experiment-2)
    3. [Experiment 3](#experiment-3)
    4. [Experiment 4](#experiment-4)
    5. [Experiment 5](#experiment-5)
8. [Conclusion](#conclusion)
9. [Future Works](#future-works)
10. [References](#references)

## Introduction
High quality labeled data is one of the reasons for the success of Deep Learning models.
But human labeling involves a lot of effort. On top of that, getting labeled data for
Biological Image Segmentation is more challenging because a human expert has to spend
hours labeling each pixel which is cumbersome. To avoid human labeling, techniques of
self-supervised learning were developed which can do away with less labeled data and
are currently in research.

The objective of this project is to implement a Self Supervised learning model on
biological data collected from LightSheet Microscope for semantic segmentation. By
using some standard image processing techniques, it can be further developed to do
instance segmentation. We expect the self-supervised learning technique which is used in
our model to perform as efficiently as a standard fully supervised model.

The report is divided as follows. First, we go over the existing solutions and discuss
potential improvements. Second, we discuss the mechanism behind self-supervised
learning and segmentation. Third, we present the dataset that is used for our segmentation
task, its dimensions, preprocessing involved, etc. Fourth, we describe the workflow for
training. Fifth we report our findings, expectations, and analysis. Lastly, we conclude the
report with further developments possible in this project.

## Literature Review
Defining a proper pretext task that is relevant to the downstream task is of utmost importance for
the performance of the model on downstream tasks. Gidaris et.al [[1]](#ref1) proposed learning
representations by predicting the rotation of an image as a pretext task. Zhang et.al [[2]](#ref2) proposed
coloring an image if a grayscale is given as a pretext task. Noorozi et.al [[3]](#ref3) created a pretext task
where the network has to solve the jigsaw puzzle of the image. Doersch et.al [[4]](#ref4) proposed a
pretext task where the network will learn by answering some contextual questions corresponding
to the query image. But the design of pretext tasks depends heavily on the domain, and it is
increasingly becoming difficult to create new, meaningful pretext tasks. Also, one could not
quantitatively analyze how a model will perform on downstream tasks, given proper training on
pretext tasks. This imposes another constraint on how to measure a good pretext task.

So, Ting et.al [[5]](#ref5) proposed SimCLR (Simple Contrastive Learning Representations) where they
used contrastive learning i.e the representations from the same sample will be forced to be closer
and the representations from different samples will be farther. Though it achieves state-of-the-art
accuracy for classification, it requires a larger batch size, more training steps, and negative pairs
to get a decent accuracy which is not feasible in resource-constrained environments. Also, the
authors pointed out that a set of image augmentations performed in the same order will improve
the efficiency but the exact reason why the specific combination yields good results is unknown.
Kaiming He et.al [[6]](#ref6) proposed MoCo (Momentum Contrast) where they maintain a dynamic
dictionary bank (as keys) and when a query is given, they perform some preprocessing and
compare the contrastive loss between the keys in the dictionary and query. Even though it
performs well, it has a slow-moving momentum encoder and has a dictionary that needs to be
updated constantly. So, Bastien et.al [[7]](#ref7) proposed BYOL (Bring Your Own Latent) where they
have two networks (online and target) that interact and learn from each other, thus avoiding the
need for negative pairs and large batch size and is also robust to random image augmentations.

Though the above models perform well in tasks like classification, they did not achieve
reasonable performance in dense prediction tasks such as segmentation. One plausible reason
might be that the above methods treat each image as an instance, while for dense tasks like
segmentation, each pixel is considered as an instance. So, PixelCL, a more recent architecture by
Zhenda et.al [[8]](#ref8) went further and introduced a pixel level loss function that is specifically
designed for segmentation tasks.

## Self-Supervised Learning
It is a form of unsupervised learning where we withhold some part of the data and tasks a
neural network to predict it from the remaining parts and these proxy tasks that we create
are called pretext tasks. There are various cognitive principles followed, out of which one
of them is called Contrastive Learning.

The idea is to find powerful representations of input data (also called embeddings). The
quality of the embedding is dependent on the downstream task (segmentation in this case)
and it should perform better upon feeding the embeddings into the segmentation model
compared to feeding direct original input into it. For Computer Vision, we generally
choose the output of the last layer before softmax in a ‘convnet’ as embedding.

The general flow can be summarized as:
1. Create multiple views of the input data using data augmentation (eg: random
cropping, resizing, color distortion, etc.)
2. Use a base encoder (say ResNet) to map these inputs to embeddings.
3. Compute a contrastive loss that describes the similarity between correlated images.
4. Repeat till the loss decrement converges.

In this project, we have used PixelCL as the backbone architecture for self-supervised
segmentation, and for a fully supervised setup, we are using plain ResNet as the
backbone architecture.

## Segmentation

Image segmentation is a computer vision task in which we label specific regions of an
image. More specifically, the goal of semantic image segmentation is to label each pixel
of an image with a corresponding class of what is being represented. In this project, we
focus only on semantic segmentation where we are not separating instances of the same
class and we only care about the category of each pixel. By using some standard image
processing algorithms, instance segmentation can be easily done on top of the semantic
segmented output. Segmentation is a precursor for several applications in biomedical
image analysis, such as population counting, cell tracking, and analyzing the
evolution/interactions among various cells.

Our goal is to take a grayscale image (height×width×1) and output a segmentation map
where each pixel contains a class label represented as an integer (height×width×1). In this
project, there are only two classes (cells vs background noise) i.e. 255 or 0 as the pixel
intensity for semantic segmentation. Individual cells are labeled using different shades of
gray (for instance segmentation).

For the architecture, we downsample (encode) our feature maps through pooling or
strided convolutions developing lower-resolution feature mappings which are learned to
be highly efficient at discriminating between classes, and then upsample (decode) the
feature representations into a full-resolution segmentation map. This is necessary as a
fully-connected neural network will be computationally expensive.

We mainly focus on FCN (Fully Convolutional Networks) architecture for the
upsampling task. FCN consists of transpose convolutional layers but since the encoder
module reduces the resolution of the input, the decoder module struggles to produce
fine-grained segmentation.

## Dataset

We focused on datasets that were collected using the light-sheet microscopy technique. In
this setup, light is not directly projected on the sample but is rather passed across the
cross-section to the sample. A camera is placed in front of the sample, perpendicular to
the light plane. Because of the fluorescence property of the sample, the photons will get
captured by the lens and we can see a particular cross-section of the sample. Datasets
collected by these methods are prone to have low SNR (because of low light conditions)
and it is very difficult to capture good information of the sample because of the
non-deterministic fluorescence property. Usually, a living sample is placed to understand
the interactions between various parts of the sample and the imaging should be very fast
to capture minute details of the sample.

We used [c.elegans dataset](https://www.nature.com/articles/nmeth.1228) published by the University of Washington for training both
self-supervised and downstream models. The ‘c.elegans dataset’ has 180 3D stacks 
(along with masks) where each stack has 35 images of size 512 x 708. So, the dimension
of the dataset is (180 x 35 x 512 x 708) i.e. (No of stacks x No of images per stack x
width x height). We took five 3D stacks for the entire project. Note that images are 16-bit.

Each image in the stack is further divided into patches of size 384 x 384 with a window
stride of 64 on both the axes. So, the final dataset is (5 x 35 x 12 x 384 x 384) i.e. (No of
stacks x No of images per stack x No of patches per image x width x height). Most of
these patches have no information in them because the organism is present only in a
portion of the image. So, it is important to remove the noisy patches before passing the
data to the model. We used the mean of all the patches as a threshold. If a patch has a
mean value less than this threshold, it will be discarded.

Thus, our final dataset can be described as:
* Training data for backbone using Self Supervised Learning (SSL) technique: 1370
image patches without their masks.
* Training data for FCN head i.e downstream segmentation model: Image patches
with their masks (training loss is computed). The number will vary as per the
experiments. See ‘Results’ for more details.
* Validation data for FCN head i.e downstream segmentation model: 37 image
patches with their masks (validation loss is computed)
* Test data for FCN head i.e downstream segmentation model: 100 image patches
(masks used only for inference)

## Workflow

Contrastive Learning works best when we have lots of unlabelled data where the model learns
useful information from the data. But how well it learns from the unlabelled data depends on the
loss function. Architectures such as SimCLR and BYOL work well for classification and
detection tasks but are not so good for learning embeddings for segmentation tasks because they
are trained for instance-level (the entire image is considered as a single data point). So, the
features extracted from PixelCL will be suitable for our downstream task i.e. making it more
suitable for tracking.

In stage 1, we use PixelCL as a backbone model, training the unlabeled data for the backbone as
shown in the figure below. In stage 2, we freeze the layers of the backbone model, attach the
FCN head to this backbone model and train the labeled data for the FCN head. Later, we
unfreeze the layers making it backbone trainable for ablation study.

![1](https://user-images.githubusercontent.com/31370694/166185974-64623ddc-446d-4ab3-894f-611d6f9d1b50.PNG)  
**Fig. 1:**  Stage 1 training using only data by Self Supervised Learning

![2](https://user-images.githubusercontent.com/31370694/166186030-9279d732-faf4-4cc0-9701-1ce84cab1de9.PNG)  
**Fig. 2:**  Stage 2 of proposed Backbone + FCN model

We shall use ‘Dice Score’ as a metric for comparing predicted segmentation masks with their
actual masks. When two sets are given, it compares the similarity between those two sets and it
works well for Image Segmentation because we can assume the true mask and predicted mask to
be a set of two.

Dice Score = 2 * |X ∩ Y| / (|X| + |Y|) where X and Y are the two sets and |.| refers to the
cardinality of a set.

## Results

We use the following terminologies for describing the various experiments performed.  
* ‘unlabeled’: Image patches without their segmentation masks.
* ‘labeled’: Image patches with their segmentation masks.
* ‘bb-frozen’: Backbone layers are frozen when training FCN.
* ‘bb-tunable’: Backbone layers are not frozen when training FCN.

Note:
1. Connected Component Labeling technique is applied on top of the semantic
segmented output to generate instance segmentation masks (different shades of
gray in the output mask correspond to different labels).
2. Every experiment is run for 10 epochs.

### Experiment 1

The goal of this experiment is to evaluate the model as we vary the amount of ‘labeled’
data passed to the FCN head in the downstream task. We are trying to determine the
minimum amount of segmentation labels needed for the model to perform on par with the
fully supervised plain ResNet backbone architecture with FCN head. So, we vary the
number of ‘labeled’ data to train the FCN head.

#### Experiment 1.1
* Train PixelCL backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-frozen) with 1233 ‘labeled’ images. (90 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head with 100 ‘unlabeled’ images.

The backbone took 2min/epoch and FCN took 1min/epoch.
Loss plots for the model and image samples are shown below.  

![3](https://user-images.githubusercontent.com/31370694/166186627-4d638ee1-b78a-4c8b-b124-c13d41e15768.PNG)  
**Fig. 3:**  Loss plots for expt 1.1

![4](https://user-images.githubusercontent.com/31370694/166186775-92f6980f-a6b7-4b82-84ba-6c5c6f6ec098.PNG)  
**Fig. 4:**  Input Image

![5](https://user-images.githubusercontent.com/31370694/166186813-900a1fa8-5f44-4411-9013-006781e9347f.PNG)  
**Fig. 5:**  Original Mask (Left) and Predicted Mask (Right)

Note: The Loss plot for the backbone (Fig.4, Left) is reducing for some experiments and
did not reduce for some. (For experiments where it was reduced, the dice score was 0.9)
So, we believe it will converge if it was run for enough epochs.

#### Experiment 1.2
* Train PixcelCL backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-frozen) with 50 ‘labeled’ images. (3.5 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

Loss plots for the model and image samples are shown below. The input image is the
same as Experiment 1.

![6](https://user-images.githubusercontent.com/31370694/166187033-106e8e33-3bd1-4e25-9c2b-64ee47c1f784.PNG)  
**Fig. 6:**  Loss plots for expt 1.2

Note: The predicted mask is not visible i.e the model didn’t predict anything reasonable.

#### Experiment 1.3
* Train PixcelCL backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-frozen) with 100 ‘labeled’ images. (7 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

Loss plots for the model and image samples are shown below. The input image is the
same as Experiment 1

![7](https://user-images.githubusercontent.com/31370694/166187171-052ed40c-3d5d-45e2-81d9-3ed0ed215a61.PNG)  
**Fig. 7:**  Predicted Mask (Left) and Loss Plot for expt 1.3

#### Experiment 1.4
* Train PixcelCL backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-frozen) with 500 ‘labeled’ images. (36.5 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

Loss plots for the model and image samples are shown below. Input image is the same as
Experiment 1
![8](https://user-images.githubusercontent.com/31370694/166187329-c8e972ee-9d83-4660-9d9b-17db367a259b.PNG)  
**Fig. 8:**  Predicted Mask (Left) and Loss Plot for expt 1.4

| Sub-experiment  | No. of labeled images | Dice Score (averaged) |
| :---            |    :----:             |          ---:         |
| 1.1             | 1233 (90%)            | 0.73                  |
| 1.2             | 50 (3.5%)             | 0.16                  |
| 1.3             | 100 (7%)              | 0.35                  |
| 1.4             | 500 (36.5%)           | 0.69                  |

### Experiment 2

* Train PixcelCL backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-frozen) with 100 ‘labeled’ images. (7 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

In this experiment, we trained the PixelCL backbone attached to FCN head using a Self
Supervised setup with backbone weights tunable. We still used 1370 ‘unlabeled’ images
for training the backbone and vary the number of ‘labeled’ images for training the FCN
head.

| Sub-experiment  | No. of labeled images | Dice Score (averaged) |
| :---            |    :----:             |          ---:         |
| 1.1             | 1233 (90%)            | 0.92                  |
| 1.2             | 50 (3.5%)             | 0.57                  |
| 1.3             | 100 (7%)              | 0.78                  |
| 1.4             | 500 (36.5%)           | 0.89                  |

### Experiment 3

In this experiment, we trained a plain Resnet backbone attached to the FCN head i.e. a
fully supervised setup with backbone weights tunable.

#### Experiment 3.1
* Train ResNet backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-tunable) with 1233 ‘labeled’ images. (90 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

![9](https://user-images.githubusercontent.com/31370694/166187804-21483d33-38f6-4651-a224-39ec628d6663.PNG)  
**Fig. 9:** Loss Plot for Resnet backbone + FCN head (Left) and Predicted Mask (Right)

#### Experiment 3.2

* Train ResNet backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-tunable) with 50 ‘labeled’ images. (3.5 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

![10](https://user-images.githubusercontent.com/31370694/166187915-16907bc4-e019-474d-9ac4-b45f47167914.PNG)  
**Fig. 10:** Loss Plot for Resnet backbone + FCN head (Left) and Predicted Mask (Right)

#### Experiment 3.3

* Train ResNet backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-tunable) with 100 ‘labeled’ images. (7 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

![11](https://user-images.githubusercontent.com/31370694/166188078-67f9f7a0-3c3f-431d-8b7e-2e6b2d940353.PNG)  
**Fig. 11:** Loss Plot for Resnet backbone + FCN head (Left) and Predicted Mask (Right)

#### Experiment 3.4
 
* Train ResNet backbone with 1370 ‘unlabeled’ images. Train FCN head
(bb-tunable) with 500 ‘labeled’ images. (36.5 percent of data labeled)
* Validate the backbone and FCN head with 37 ‘labeled’ images.
* Test backbone and FCN head 100 ‘unlabeled’ images.

![12](https://user-images.githubusercontent.com/31370694/166188166-270d018b-fc67-4bee-a2c9-70b48a7c73c4.PNG)  
**Fig. 12:** Loss Plot for Resnet backbone + FCN head (Left) and Predicted Mask (Right)

| Sub-experiment  | No. of labeled images | Dice Score (averaged) |
| :---            |    :----:             |          ---:         |
| 1.1             | 1233 (90%)            | 0.93                  |
| 1.2             | 50 (3.5%)             | 0.52                  |
| 1.3             | 100 (7%)              | 0.73                  |
| 1.4             | 500 (36.5%)           | 0.89                  |

### Experiment 4

This experiment is similar to Experiment 3 with backbone weights frozen. We still use
1370 ‘unlabeled’ images for training the backbone and vary the number of ‘labeled’
images for training the FCN head.

| Sub-experiment  | No. of labeled images | Dice Score (averaged) |
| :---            |    :----:             |          ---:         |
| 1.1             | 1233 (90%)            | 0.87                  |
| 1.2             | 50 (3.5%)             | 0.46                  |
| 1.3             | 100 (7%)              | 0.65                  |
| 1.4             | 500 (36.5%)           | 0.79                  |

### Experiment 5

In this experiment, we trained PixelCL backbone + FCN head using Self Supervised
Learning with backbone weights tunable. This time we vary the number of ‘unlabeled’
images used for training the backbone along with the number of ‘labeled’ images for
training the FCN head to obtain a comprehensive analysis.

![13](https://user-images.githubusercontent.com/31370694/166188722-826ddab1-012b-4a8b-81a6-41ea7307b2b8.PNG)  

# Conclusion

We observed that a model trained with the SSL technique achieves a dice score of about
0.73 with only 10 epochs and less unlabeled data (we have 10000+ images of unlabeled
data, but chose a small sample size). Even when the number of labeled samples is
reduced to 35% (500 samples), it achieved a reasonable dice score of around 0.69 which
is very close to the performance using the entire data. It struggled to predict well using
only 50 and 100 samples. With UNet, we achieved a dice score of 0.56. In the ablation
study (Experiment 4), we observed that the FSL achieved more dice score compared to
SSL.

So, if we have more unlabeled data, and few labeled data (in our case it’s around 500), we
can get good performance using techniques from SSL.

We can increase the dice score by the following methods:
1. An increasing number of epochs
2. Increasing training data (both labeled and unlabeled). We have data, but due to
time constraints, we chose a small sample size.
3. Modeling a better FCN head. Right now, we are doing a couple of transposed 2D.
We can replace the FCN head with the UNet head i.e Resnet backbone + UNet
head might perform better compared to our present Resnet backbone + FCN head

# Future Works

The purpose of the Self Supervised Learning technique is to reduce manual annotations
for datasets. So, we collected unlabelled data (as shown in Figure. 11) of zebrafish (an
organism) from Morgridge lab, University of Wisconsin Madison to examine whether a
pre-trained model in a similar domain (c.elegans dataset in this particular case) can
generalize well.

![14](https://user-images.githubusercontent.com/31370694/166188821-6f17c7f2-85a2-415a-8f1a-ad4d33ebc73b.PNG)  
**Fig. 13:** Example sample image (The sample is a Zebra Fish). The data is collected using
LightSheet Microscopy and each image is of size 2048 x 2048

When we did preprocessing and passed this image to our model, it did not generalize
well. While this data is a bit more complex compared to the dataset we used, further
investigation will be carried in future on the potential reasons of why the model did not
generalize well and work on alternative solutions will be done.

One reason we can come up with is the 16 bit representation of the image. When we
divided the original 2048 x 2048 image into patches of 384 x 384 and computed the mean
of all the patches (to be used as threshold), it turned out to be around 130 (But the
c.elegans dataset we worked on is a 8 bit input image, it’s threshold is around 30). So,
maybe because of this difference in the input domain space, the model failed to
generalize well.

# References

1. <a name="ref1"></a> [Unsupervised Representation Learning By Predicting Image Rotations](https://arxiv.org/abs/2011.10043)
2. <a name="ref2"></a> [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
3. <a name="ref3"></a> [Unsupervised Learning of Visual Representations by Solving Jigsaw](https://arxiv.org/abs/1603.09246)
4. <a name="ref4"></a> [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192.pdf)
5. <a name="ref5"></a> [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
6. <a name="ref6"></a> [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf)
7. <a name="ref7"></a> [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs//2006.07733)
8. <a name="ref8"></a> [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs//2011.10043)
