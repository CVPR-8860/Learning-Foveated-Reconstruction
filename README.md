# Learning-Foveated-Reconstruction

This repository is provided as a supplementary material to the "Learning Foveated Reconstruction to Preserve Perceived Image Statistics" paper. It includes the following directories:
1. `Calibrated VGG metric` - contains a MATLAB script used for measuring the predicted detection rate between two images shown at a given eccentricity. This metric is explained in Section 5.2 in the paper.
2. `GAN` - allows to generate the images based on all the methods explained in the paper in Section 5. We provide the pretrained GAN along with test images. We also provide an option of training the metric on the user-provided data.
3. `demo` - shows a detailed comparison of images generated using different training procedures, explained in the paper Section 5. The demonstration can be started by launching `reconstructions.html` in a browser. It is fully offline and can be started without internet connection. We have tested it using Firefox and Chrome.
