This repository contains an implementation of the following paper:

**FCA-Net: Adversarial Learning for Skin Lesion Segmentation Based on Multi-scale Features and Factorized Channel Attention**

**Abstract:** Skin lesion segmentation in dermoscopic images is still a challenge due to the low contrast and fuzzy boundaries of lesions. Moreover, lesions have high similarity with the healthy regions in terms of appearance. In this paper, we propose an accurate skin lesion segmentation model based on a modified conditional generative adversarial network (cGAN). We introduce a new block in the encoder of cGAN called factorized channel attention (FCA), which exploits both channel attention mechanism and residual 1-D kernel factorized convolution. The channel attention mechanism increases the discriminability between the lesion and non-lesion features by taking feature channel interdependencies into account. The 1-D factorized kernel block provides extra convolutions layers with a minimum number of parameters to reduce the computations of the higher-order convolutions. Besides, we use a multi-scale input strategy to encourage the development of filters which are scale-variant (i.e., constructing a scale-invariant representation). The proposed model is assessed on three skin challenge datasets: ISBI2016, ISBI2017, and ISIC2018. It yields competitive results when compared to several state-of-the-art methods in terms of Dice coefficient and intersection over union (IoU) score.

**Proposed Factorized Channel Attention (FCA) Block:**

![PDFtoJPG me-1 (1)](https://user-images.githubusercontent.com/18607766/59305807-3d6ec580-8c9b-11e9-8160-16d44e5ea8e1.jpg)

**Architecture of the Proposed Adversarial Network**

![skin_new_model](https://user-images.githubusercontent.com/18607766/59305103-d7357300-8c99-11e9-923a-9c09ef49a210.png)

**Intermediate Layer Visualization with different configuration of the proposed FCA-Net**

![PDFtoJPG me-1](https://user-images.githubusercontent.com/18607766/59305588-d2bd8a00-8c9a-11e9-9ce8-3a26a383e1f1.jpg)
