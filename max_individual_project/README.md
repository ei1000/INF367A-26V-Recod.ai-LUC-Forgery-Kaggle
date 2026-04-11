# README.md

_This readme contains documentation for the implementation._

## Introduction and motivation

This implementation of Deep PatchMatch is based on the paper "Image Copy-Move Forgery Detection via Deep PatchMatch and Pairwise Ranking Learning" (Li et. al., available at https://arxiv.org/pdf/2404.17310).

The implementation is tailored towards performance in the kaggle competition "Recod.ai/LUC - Scientific Image Forgery Detection" (https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/data). Because of this, there are a number of changes between my implementation and the paper version.

Most significantly, the Kaggle competition involves single-channel CMFD. This is a prediction task where the model aims to generate a binary pixel map of the same size of the image

This means that I did not implement the components of the architecture which are related to three-channel CMFD (where the ). It is possible that

## Architecture and code structure

The architecture of the project is made to be as consistent as possible with the upper half of the architecture presented in the paper.

[!TODO: image]

One important difference is that although my version also can be configured to be end-to-end differentiable, I found it was more practical not to do so, insteading using a pretrained CNN for feature extraction.

### Feature extractors

Inside `/feature_extractors` are the components utilized in the **Feature Extraction 1** part of the network.
In this implementation, the most practical choice is to use a frozen pretrained extractor such as the ones that are options in `cnn_feature_extractor.py` (VGG16 and RESNET18).
I also experimented with using Meta's _dinov2_vitb14_ model in `dino_feature_extractor.py`, but found that it was too memory-intensive to be safely utilized, particularly as my implementation batches and performs concurrent operations on . One option is to have a layer compressing Dino's output to a lower dimensionality, but then this compression layer would have to be trained, and

Although the paper does not mention specifically whether the CNN feature extractor is pretrained or not, my interpretation is that it most likely was not, as this would be consistent with the claim that the architecture is end-to-end differentiable.

In addition to the CNN feature extractor, the architecture features a parallell zernike feature extraction process based on zernike polynomials.

My implementation may not be fully consistent with the one supplied in the architecture at this point. The reason is that I through my own
It is not explained in the paper why they utilise this formula. Perhaps it a simplification that worked well in practice, or perhaps it is an implementation error.

```
for s in range(m + 1):
    c = (-1)**s * factorial(p - s) / (
        factorial(s) *
        factorial((p + abs(q)) // 2 - s) *
        factorial((p - abs(q)) // 2 - s)
    )
    R += c * rho**(p - 2*s)
```

### Deep Cross-Scale PatchMatch

When I implemented the Deep Cross-Scale PatchMatch algorithm as described in the paper, I had some issues with pixel offsets settling by finding local optima within a close vicinity. Essentially, as the features extracted within a local area are quite similar, it was hard for the algorithm to explore beyond this area. To rectify this, I implemented a method that re-randomized any offsets which were within a close vicinity to the pixel itself. As there is no mention of this in the paper, I am unsure whether my implementation diverges at this point or not.

### Multi-Scale Dense Linear Fitting and Prediction

Although this too was hard to interpret, the interpretation I decided on was that the author's implementation only used the CNN offsets to calculate DLF error maps. However, as I was getting poor

## Challenges

Although the network performed decently with regards to many metrics, it was very challenging to figure out a consistent way to have the network achieve a good OF1 score, which is the competition metric (https://www.kaggle.com/code/metric/recodai-f1/). This score is particularly punishing for two main reasons:

1. If even a few pixels are predicted to be forged on an authentic image, a score of 0 is immediately given.
2. It highly values consistent "blob"-like predictions, and predicting the correct amount of blobs. For instance, take the example of an image where there is one true copied object (source and target). Even if our model perfectly predicts this object (both source and target) if it generates say 5 other small blobs (even if they are only a few pixels each), the resulting of1 score has a maximum value of 1 \* (1/5) = 0.2.

In many of the runs, this was a major issue, with the model predicting way too many small regions on both authentic and forged images, even though it often also detected the true forged regions. In other words: recall was high, but it had poor precision.

A method that slightly helped this was applying post-processing operations of filtering out collections of pixels below a certain size. However, this is not a sure-fire fix, as there are also examples of copied regions in the dataset which are quite small. (example: train/forged/10.png).

Although it is hard to diagnose exactly what leads to this behaviour, through multiple runs and by comparing the two different branch, I believe this comes down in large part to a mistmatch between the architecture and the given dataset. The PatchMatch architecture relies mainly on finding groups of pixels which have high feature similarity. In the paper, it trains on a training set generated from MS COCO applying a copy-move with rotation and scaling operation to a random area of the image. The module is tested on a similarly generated synthetic dataset as well as the CASIA CMFD, CoMoFoD and CMH datasets.

All of these datasets feature "realistic" images. By this I mean that they are photos of nature or real life. Although the kaggle competition dataset does contain some images of this class, it also includes numerous examples which are graphics or figures featuring a background of only one consistent colour. It also includes images where the background is all very dark.

I believe that this difference in the backgrounds of images is highly significant in explaining the architectures performance on this dataset.

To align my experiments more with the paper and to test out some of these differences in image qualities, I also included the CASIA CMFD dataset in my training set. This choice was also inspired by the conception that CMFD is a general task where it is always helpful to have more data. In retrospect, this may have been a poor decision, as the CMFD task may have some inherent differences based on the class of images, as described above. In addition, including more data led to increased training times.

## Improvements given more time

Because of time limits and the complexity of the task I was not able to carry out all potential improvement points that could have led to a better model.

- Background modification in the scientific images - for instance, using a model/technique for automatically detecting background, and then adding noise to it in order to make the background less consistent.
  - The idea of this would be to cause less false positives from the PatchMatch module. When all of the background has a very similar colour and characteristics, the risk is high that we create false positives.
- Using a larger amount of data or more varied data.
- Simply running for more epochs.
- Further optimising the PatchMatch module, which was a major bottleneck in training the network. This would allow for more efficient running for a larger amount of epochs. (This part of the network is already optimised quite a lot and with some LLM assistance to make the runtime even feasible, but perhaps there is some other, better way to implement it.)

## Other inconsistencies

Unfortunately, many parameters related to the training process are neither mentioned by Li et. al. nor documented in code. This means that I had to do some experimentation on my own, leading to a structure that may have significant differences from the version Li et. al. used to achieve their results.

Although the network in theory is fully differentiable, propagating gradients throughout the whole PatchMatch module was too much to handle for my GPU memory concurrently with all other operations. Therefore, I used a frozen pretrained CNN containing the VGG16 weights (https://arxiv.org/abs/1409.1556).

## Technical details
