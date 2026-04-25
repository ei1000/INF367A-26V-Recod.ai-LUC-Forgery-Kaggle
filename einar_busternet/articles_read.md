https://medium.com/@chawthirisan/the-10-most-important-loss-functions-in-deep-learning-6d998ff3ea87

https://arxiv.org/html/2307.02694v3#S1

For looking at different loss functions relevant to the problem.

Google searches and forums also supports DICE being the go to for pixel overlap. BCE standard for classification labeling. Combining them could give a better loss function than pure BCE which I have used till now. 

https://link.springer.com/article/10.1186/s12880-026-02245-y

Hosseini and Baghshah, "Dilated Balanced cross entropy loss for medical image segmentation" (2026). Useful because it summarizes common imbalance strategies for segmentation: re-sampling, region-based losses, and loss weighting. It explicitly notes that Dice+CE/BCE style compound losses are common, with Dice helping imbalanced/small-object overlap and CE giving smoother gradients.

https://arxiv.org/abs/1810.07842

Abraham and Khan, "A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation" (2018). Useful because Tversky/focal Tversky is aimed at imbalanced segmentation and lets us bias the loss toward recall when false negatives dominate.

https://doi.org/10.1016/j.media.2021.102035

Ma et al., "Loss odyssey in medical image segmentation" (2021). Useful as a broad comparison paper for segmentation losses. Supports testing compound/region-based losses instead of relying only on BCE/CE for heavily imbalanced masks.
