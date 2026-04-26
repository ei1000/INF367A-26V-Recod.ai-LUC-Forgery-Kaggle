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

https://openaccess.thecvf.com/content_ECCV_2018/papers/Rex_Yue_Wu_BusterNet_Detecting_Copy-Move_ECCV_2018_paper.pdf

Wu, Abd-Almageed, and Natarajan, "BusterNet: Detecting Copy-Move Image Forgery with Source/Target Localization" (ECCV 2018). Architecture reference. Mani-Det and Simi-Det use VGG16 feature extractors producing 16x16x512 features for 256x256 inputs. Their mask decoder alternates BN-Inception and BilinearUpPool2D four times to restore full resolution, ending in a 256x256x6 decoder feature tensor. The binary branch classifier is a single 3x3 Conv2D + sigmoid. Fusion concatenates both decoder feature maps, applies BN-Inception 3@[1,3,5], and predicts a 3-class mask. Training is stage-wise: branch auxiliary training, frozen-branch fusion training, then end-to-end fine-tuning.

https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/datasets/decoders.py

Meta DINOv2 repository, `dinov2/data/datasets/decoders.py`. This file is not a model decoder or segmentation head; it contains dataset/sample decoding helpers for image bytes, TIFF/X-channel arrays, and channel selection. Useful negative result: it should not guide our Mani/Simi/Fusion architecture.

https://rnagara1.medium.com/decoding-dinov2-next-gen-computer-vision-with-metas-breakthrough-model-and-integration-into-eef5eae53f86

Rajath Nag Nagaraj, Medium overview of DINOv2 (2024). Light secondary source. Useful only as a quick reminder that DINOv2 is often used as a strong frozen feature backbone with task-specific heads. Not a primary architecture source for segmentation decoders.

https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/dpt

Hugging Face documentation for DPT, based on "Vision Transformers for Dense Prediction". Useful because DPT is a modern ViT dense-prediction pattern: take tokens from several transformer stages, reassemble them into image-like feature maps at different resolutions, and progressively fuse them with a convolutional decoder. This supports a more principled next BusterNet-DINO step than only widening grid convs: use multiple DINO intermediate layers and progressive feature fusion/upsampling for small-mask detail.

https://papers.nips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html

Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (NeurIPS 2021). Useful counterpoint to heavy decoders. SegFormer argues that transformer segmentation can work well with a lightweight decoder that aggregates multi-layer features, avoiding overly complex decoder blocks. For our project this supports multi-layer DINO feature aggregation over blind channel growth.

https://arxiv.org/abs/2510.00585

Sajjad, Shaukat, and Mir, "U-DFA: A Unified DINOv2-Unet with Dual Fusion Attention for Multi-Dataset Medical Segmentation" (arXiv 2025). Directly relevant to DINOv2 + medical segmentation. They keep DINOv2 frozen and use a U-Net-style encoder-decoder with local/global fusion adapters that inject CNN spatial features into DINOv2 blocks. Key lesson: for medical segmentation, frozen foundation features benefit from explicit spatial/local feature fusion, not only a final shallow head.

https://www.sciencedirect.com/science/article/pii/S1361841524002056

Chen et al., "TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers" (Medical Image Analysis 2024). Useful medical-segmentation architecture source. The paper emphasizes U-Net style encoder/decoder design, transformer attention, and notes that decoder-side attention/coarse-to-fine refinement helps small targets such as tumors. This supports investigating progressive/coarse-to-fine decoding for our small/tiny forged masks.

https://github.com/isi-vista/BusterNet

Official BusterNet repository. Confirms that the released project follows the paper's
two-branch Mani/Simi design followed by a fusion module. Useful as a code/provenance
source: our project is still a BusterNet adaptation, but DINO replaces the VGG feature
extractors and the competition objective motivates binary union fusion.

https://github.com/imagecbj/A-serial-image-copy-move-forgery-localization-scheme-with-source-target-distinguishment

Chen et al. implementation/repository for "A Serial Image Copy-Move Forgery Localization
Scheme With Source/Target Distinguishment" (IEEE TMM 2020). Important BusterNet successor.
Their stated criticism of BusterNet matches our concern: Simi-Det uses single-level,
low-resolution VGG features and both branches must locate regions correctly. They improve
copy-move similarity detection with higher-resolution/multilevel ideas, atrous
convolution, ASPP, attention, and a serial source/target distinguishment stage. Takeaway:
for BusterNet-like CMFD, stronger similarity decoding and multiscale context are known
directions, not blind parameter growth.

https://pmc.ncbi.nlm.nih.gov/articles/PMC12941880/

Lu and Zhang, "Robust Detection and Localization of Image Copy-Move Forgery Using
Multi-Feature Fusion" (J. Imaging 2026). Recent CMFD fusion paper. It argues that
existing CMFD models under-use complementary features during fusion and often decode by
simple pixel classification without cross-layer aggregation or local/global attention.
Their MFFNet uses dual-domain features, a two-stage feature fusion module, and a
lightweight multi-layer decoder that aggregates hierarchical features before producing
the localization map. Takeaway: for copy-move forgeries, modern fusion is usually
multistage/hierarchical, not just concat + one classifier.

https://www.nature.com/articles/s41598-025-97779-6

Zhang et al., "Medical image segmentation by combining feature enhancement Swin
Transformer and UperNet" (Scientific Reports 2025). Relevant domain analogue. UPerNet
style decoding is used because medical targets have multi-scale organ/tissue structure.
Takeaway: for medical segmentation, pyramid-style fusion of features at different scales
is common when targets vary strongly in size.

https://www.nature.com/articles/s41598-024-84685-6

Zhu et al., "A hybrid attention multi-scale fusion network for real-time semantic
segmentation" (Scientific Reports 2025). General segmentation fusion source. It states
the usual reason for feature fusion: combine shallow local/edge information with deeper
semantic/global information, because features from one layer cannot capture everything.
Takeaway: if we add DINO intermediate features later, fusion should align channels and
resolutions before concatenation rather than naively mixing incompatible maps.

https://biodatamining.biomedcentral.com/articles/10.1186/s13040-023-00320-6

Jiang et al., "iU-Net: a hybrid structured network with a novel feature fusion approach
for medical image segmentation" (BioData Mining 2023). Useful medical segmentation
fusion analogue. The paper frames U-shaped skip/fusion structures as a way to retain
local spatial detail while adding global context. Takeaway: our DINO-only branch heads
may need explicit spatial refinement/fusion because frozen global features alone can be
too coarse for tiny masks.

https://www.sciencedirect.com/science/article/pii/S0925231225010239

MFF-Net, "A multi-view feature fusion network for generalized forgery image detection"
(Neurocomputing 2025). General forgery detection, not CMFD localization, but useful
because it uses multi-view spatial/frequency/texture features and adaptive fusion for
generalization. Takeaway: image forensics often benefits from complementary-view fusion.
For this project we keep DINO and source/similarity branches rather than adding frequency
branches now, but adaptive or attention-weighted branch fusion is a reasonable later
ablation.
