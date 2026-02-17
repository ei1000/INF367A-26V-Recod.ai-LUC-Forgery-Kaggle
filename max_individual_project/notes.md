## Paper link

https://arxiv.org/pdf/2404.17310

## Other useful resources

## CNN feature extractor:

Our backbone architecture comprises five convolution blocks, each consisting of a convolution layer, a BatchNorm layer, and a
ReLU layer. At the end of the backbone, there is a resizing
layer to rescale the feature maps to the same dimensions of
H × W ×c, where c represents the number of feature channels,
empirically set to 32
