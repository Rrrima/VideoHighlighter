# VideoHighlighter
course project for SI671
---

With massive amounts of video data appearing in people’s normal lives, key frames extraction and summary are of great significance to content-based video retrieval and video navigation. 
    
We are trying to find a solution to help users quickly access frames of interest with preferred granularity. We implemented multiple image mining and modeling techniques including constructing explainable feature maps, identifying latent factors, clustering and regression.

We achieved the prediction accuracy of 0.0716 MAE for superframe scoring and 0.0684 for single frame scoring with a range of 0-1. Besides interest scores prediction, the final system is able to identify superframes and generate summaries with customizable verbosity. In the end, We give a thorough analysis of factors that account for visual interest and give graphic illustrations of Video-Highlighter system . 
---

This code contains all classes related to the system. However, there is no dataset and the pretrained ResNet-50 model. 

Please refer to 

- [SumMe](https://gyglim.github.io/me/vsum/index.html) for the dataset.
- [PoolNet](https://github.com/backseason/PoolNet) for the pre-trained model


