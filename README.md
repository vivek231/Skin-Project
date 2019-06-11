FCA-Net: Adversarial Learning for Skin Lesion Segmentation Based on Multi-scale Features and Factorized Channel Attention

Abstract: Skin lesions segmentation in dermoscopic images is still a challenge due to the low contrast and fuzzy boundaries of lesions. Besides, lesions have high similarity in terms of appearance, to healthy regions. In this paper, we propose an accurate skin lesion segmentation model based on a modified conditional Generative Adversarial Network (cGAN). To promote coarse to fine features that are more related to skin lesions, the input layer of the proposed network is set for processing multi-scale images. In addition, we introduce a novel layer used the encoder of the cGAN, called Factorized Channel Attention (FCA). That layer integrate a channel attention block and a residual 1-D kernel factorized convolution. The channel attention block increases the discriminability between the lesion and non-lesion features by taking into account feature channel interdependencies. The 1-D factorized kernels provide extra convolutional layers with a minimal set of parameters and a residual connection that minimizes the impact of image artifacts and irrelevant objects.  The proposed model is assessed on three skin challenge datasets: ISBI2016, ISBI2017, and ISIC2018. It yields the best results among several state-of-the-art methods in terms of dice index and Intersection Over Union (IoU) score, achieving 93.9% and 87.6% with ISBI2016 dataset, 88.3% and 78.9%  with ISBI2017 dataset, respectively, and IoU score of 77.2% with validation set of ISIC2018 dataset.

Proposed Factorized Channel Attention FCA Block:

![PDFtoJPG me-1 (1)](https://user-images.githubusercontent.com/18607766/59305807-3d6ec580-8c9b-11e9-8160-16d44e5ea8e1.jpg)

Architecture of the Proposed Adversarial Network

![skin_new_model](https://user-images.githubusercontent.com/18607766/59305103-d7357300-8c99-11e9-923a-9c09ef49a210.png)

Intermediate Layer Visualization with different configuration of the proposed FCA-Net

![PDFtoJPG me-1](https://user-images.githubusercontent.com/18607766/59305588-d2bd8a00-8c9a-11e9-9ce8-3a26a383e1f1.jpg)
