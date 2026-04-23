_This document contains more technical documentation and information for the implementation than the README._

## Introduction and motivation

This implementation of Deep PatchMatch is based on the paper "Image Copy-Move Forgery Detection via Deep PatchMatch and Pairwise Ranking Learning" (Li et al., available at https://arxiv.org/pdf/2404.17310).

This implementation is an adaptation of Deep PatchMatch for the Kaggle competition "Recod.ai/LUC - Scientific Image Forgery Detection" (https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/data). Because the competition task differs from the setting in the paper, I made several intentional design changes to better fit the dataset, output format, and practical resource constraints. These changes include the use of DINOv2 and ResNet18 models, in addition to pixel mask post-processing techniques.

Most significantly, the Kaggle competition involves single-channel CMFD. This is a prediction task where the model aims to generate a binary pixel map of the same size as the image, where a 0-value indicates an authentic pixel, whereas a 1-value indicates a forged pixel. A forged pixel is defined as one that is either part of the source region, where the object is copied from, or the target region, where the object is copied to.

The architecture proposed in the paper supports three-channel CMFD, where instead of a binary pixel map, the model predicts for each pixel whether it is part of the background, source, or target. I did not implement the components related to performing three-channel CMFD.

In addition, many training details are not specified by Li et al. and were not available in code, so I had to make independent implementation choices for several parts of the pipeline. As a result, this project should be understood as a task-specific adaptation of the method rather than a strict line-by-line reproduction.

## Architecture and code structure

The architecture of the project is made to be as consistent as possible with the upper half of the architecture presented in the paper.

![Deep Cross-Scale PatchMatch Architecture](architecture.jpg)

One important overall difference is that I chose not to keep the full pipeline end-to-end differentiable during training. Instead, I used a pretrained frozen CNN for feature extraction, which was a more practical design given the available hardware.

### High-Level overview

The final output of the model is a post-processed argmax fusion of the masks generated from SEUNet (`se_u_net.py`, running DINOv2 feature extraction + decoder) and the mask generated from PatchMatch (`cnn_feature_extractor.py` + `zernike_feature_extractor.py` $\to$ `pixel_propagator.py` $\to$ `multi_scale_dlf.py` $\to$ `dlfdecoder.py`). This is orchestrated in the file `pipeline.py`, which is necessarily quite verbose due to the large number of configurable settings in `pipeline_config.py` and the extensive logging and saving operations performed.

Following is a more detailed description of the code and structure.

### Feature extractors

Inside `/feature_extractors` are the components utilized in the **Feature Extraction 1** part of the network. These include a CNN feature extractor and a Zernike feature extractor. My implementation applies both feature extractors to each image at the scalings $0.75, 1,$ and $1.5$, consistent with the paper.

#### CNN

In this implementation, the most practical choice is to use a frozen pretrained extractor such as those available in `cnn_feature_extractor.py` (VGG16 and ResNet18).

I also experimented with using Meta's _dinov2_vitb14_ model, but found that it was too memory-intensive to be safely utilized in this branch, particularly since my implementation batches and performs concurrent operations on different image scalings. Note, however, that DINOv2 is used in the independent SEUNet branch.

Although the paper does not explicitly state whether the CNN feature extractor is pretrained, it is reasonable to assume that it was not, given that the architecture is described as fully end-to-end differentiable. In practice, however, using an untrained feature extractor led to poor performance and proved difficult to optimize, largely because it required gradient propagation through the entire PatchMatch module. While the network is theoretically fully differentiable, propagating gradients through the full PatchMatch pipeline was too demanding for available GPU memory when combined with other operations.

To address this, a frozen pretrained CNN based on VGG16 was used as the feature extractor. This modification improved stability and performance and enabled the use of a hard argmax operation instead of the differentiable soft argmax controlled by the $\beta$ parameter described in the paper.

#### Zernike

In addition to the CNN feature extractor, the architecture features a parallel Zernike feature extraction process based on Zernike polynomials.

In the Zernike branch, I intentionally used a different formulation from the one presented in the paper. This choice was made after comparing the formula in the paper with the formulation used in the cited source and other standard references, which appeared better justified mathematically.

In the paper, the following formula is presented for the orthogonal radial polynomial $R_{p,q}(\rho)$:
$R_{p, q}(\rho) = \sum_{s=0}^{(1 - |p|)/2} \frac{(-1)^s[(1-s)!]\rho^{1-2s}}{s!(\frac{1+|p|}{2}-s)!(\frac{1-|p|}{2}+s)!}$

However, the referenced source uses:
$R_{p, q}(\rho) = \sum_{s=0}^{(p - |q|)/2} \frac{(-1)^s[(p-s)!]\rho^{p-2s}}{s!(\frac{p+|q|}{2}-s)!(\frac{p-|q|}{2}+s)!}$
(rewritten using variables $p, q$ instead of $n, m$)

(see: https://www.researchgate.net/profile/Simon-Liao/publication/6534347_Accurate_Computation_of_Zernike_Moments_in_Polar_Coordinates/links/0c960524df7a3c3257000000/Accurate-Computation-of-Zernike-Moments-in-Polar-Coordinates.pdf)

These forms are also presented here: https://mathworld.wolfram.com/ZernikePolynomial.html. Another reason this formulation seemed more logical is that it explicitly depends on $q$.

It is not explained in the paper why a different formula is used. This may be a simplification that worked in practice or an implementation detail. Without further clarification, I found it more reasonable to use the formulation supported by other sources.

This implementation uses a maximum order of ZM equal to 5, resulting in a 12-dimensional feature map for each image scaling.

As the Zernike kernels are the same for each run, they are precomputed and cached.

### Deep Cross-Scale PatchMatch

![PatchMatch Architecture](patchmatch_architecture.jpg)

The PatchMatch architecture is implemented in `cross_scale_patchmatch/pixel_propagator.py`.

During implementation, I observed that pixel offsets often converged to local optima within a small neighborhood. Since features in local regions are quite similar, the algorithm struggled to explore more distant matches.

To address this, I introduced a re-randomization step for offsets that remained too close to their original pixel. This improves exploration and reduces convergence to poor local optima.

The paper does not specify the number of iterations used. In this implementation, 24 iterations were necessary to obtain good offsets.

Here is an example of generated offsets:
![PatchMatch Offsets](offsets_example.png)

### Multi-Scale Dense Linear Fitting and Prediction

The `prediction/multi_scale_dlf.py` file creates predicted error maps based on pixel offsets from `pixel_propagator.py`. These are then passed to the DLFDecoder in `dlfdecoder.py` to obtain the final mask.

While the paper primarily relies on CNN offsets, this implementation also incorporates Zernike offsets, which improved empirical performance on the competition task.

### Loss functions

In addition to the two Dice loss functions used in the paper for $M$ and $M_t$ ($M$ = final merged mask, $M_t$ = SEUNet predicted mask), I added BCELoss, as it was used in the main group implementation. I also added a Dice loss term for $M'$ and an additional penalty for false positives to reduce overprediction.

### Post-processing steps

The pipeline applies post-processing steps using `pixelmaputil_mask.py`. These steps are important for achieving good performance on the competition metric.

In particular, the `remove_small_components` function has a large impact on reducing false positives and uses `ndimage.label` to identify connected components.

The applied pipeline is:
gaussian blur $\to$ opening $\to$ removal of small components $\to$ closing $\to$ fill components.

However, removing small components can also eliminate true positives, as some copied regions in the dataset are small.

## Challenges

Although the network performed reasonably well on several metrics, achieving a good OF1 score (https://www.kaggle.com/code/metric/recodai-f1/) was challenging.

This metric is particularly strict for two reasons:

1. If even a few pixels are predicted as forged on an authentic image, the score is 0.
2. It strongly favors consistent "blob-like" predictions with the correct number of regions.

For example, if an image contains one true copied object, and the model correctly predicts it but also produces 5 small false blobs, the maximum score becomes $1 \cdot (1/5) = 0.2$.

In many runs, the model predicted too many small regions on both authentic and forged images. While it often detected the true forged regions, recall was high but precision was low.

This behavior is likely due to a mismatch between the architecture and the dataset. PatchMatch relies on identifying groups of pixels with similar features. In the paper, training is performed on natural images (e.g., MS COCO) with synthetic copy-move transformations, and evaluation is done on similar datasets.

In contrast, the Kaggle dataset includes many scientific images with uniform or very dark backgrounds. These characteristics increase the likelihood of false matches.

To explore this further, I also included the CASIA CMFD dataset in training. However, for the final evaluation, only the competition dataset is used.

## Improvements given more time

Because of time constraints and the complexity of the task, several potential improvements were not explored:

- Background modification in scientific images (e.g., adding noise to uniform backgrounds to reduce false positives)
- Using more or more diverse data
- Training for more epochs
- Further optimizing the PatchMatch module (a major computational bottleneck)
- Exploring alternative post-processing techniques

## Disclaimer: Use of LLM and group project resources

This project was completed with some assistance from LLMs, and parts of the implementation are inspired by the shared group project. In particular, `dino_feature_extractor.py` and `pixelmaputil_mask.py` are heavily inspired by group project versions.

### LLM usage

- Optimization of pixel-wise operations using vectorized approaches (e.g., `torch.roll`, `.repeat`)
- Helper methods for tensor handling and error checking (e.g., `_as_batched_errors()` in `DLFDecoder`)
- Improvements to DLF computation and helper methods (including `box_sum()`)
- Assistance with dataset splitting logic
- Helper functions for logging and saving pipeline outputs
