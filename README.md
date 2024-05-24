# MDCR-Net: Multi-scale Decomposed Convolutional Residual Network with Strip Pooling Attention for Nighttime Lane Detection
![图2](https://github.com/JSJ515-Group/MDCR-Net/assets/113502037/a6d2c619-c985-4d05-bd73-a5aff5d3a5a7)
Abstract—Lane detection is crucial tasks in autonomous driving. Due to uneven lighting and occlusion in night scenes, there are still great challenges in nighttime lane line detection. Some prior methods have utilized additional annotated information specific to nighttime environments for training networks. However, acquiring such annotations for nighttime images is not only costly but also challenging to ensure label consistency. In this work, we propose a Multi-scale Decomposed Convolutional Residual Network (MDCR-Net) for nighttime lane detection, which is an end-to-end model for mining both global and local lane information. Specifically, we use Multi-scale Decomposition of Convolutional Residuals Module (MDCRM) to extract richer semantic features. The extracted multi-scale features are then fed into the Strip Pooling Attention (SPA) module, which estimates the global context information of the obscured location by predicting the confidence scores for the horizontal and vertical directions of the lanes in the image. This approach eliminates the need for additional annotated data and effectively mitigates occlusion issues. Extensive experiments on CULane-Night, a nighttime dataset extracted by CULane, show that MDCR-Net outperforms state-of-the-art semantic segmentation models, with F1-measure reaching 77.72% (an improvement of 6.65%) and IoU reaching 66.07% (an improvement of 8.29%). Simultaneously, achieve efficient lane detection in images at 67 frames per second (FPS) on embedded devices.
# Results:
![image](https://github.com/JSJ515-Group/MDCR-Net/assets/113502037/18c1b043-26bd-4c1c-bdd7-188202ee606f)
![image](https://github.com/JSJ515-Group/MDCR-Net/assets/113502037/48121ac9-266b-4b4e-9272-8f16b191d9bc)
