# ProbFace


This is a demo code of training and testing [ProbFace] using Tensorflow.
ProbFace is a reliable Probabilistic Face Embeddging (PFE) method.
The representation of each face will be an Guassian distribution parametrized by (mu, sigma), where mu is the original embedding and sigma is the learned uncertainty. Experiments show that ProbFace could
+ improve the robustness of PFE.
+ simplify the calculation of the multal likelihood score (MLS).
+ improve the recognition performance on the risk-controlled scenarios.


## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Usage

### Preprocessing

Download the MS-Celeb-1M dataset from [insightface](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and decode it using [this code](https://github.com/deepinsight/insightface/blob/master/recognition/data/rec2image.py)

### Training
1. Download the base model [ResFace64](https://drive.baidu.com/open?id=1MiC_qCj5GFidWLtON9ekClOCJu6dPHT4) and unzip the files under ```log/resface64```.

2. Modify the configuration files under ```configfig/``` folder.

4. Start the training:
    ``` Shell
    python train.py configfig/resface64m_msarcface.py
    ```

### Testing
+ **Single Image Comparison**
    We use LFW dataset as an example for single image comparison. Make sure you have aligned LFW images using the previous commands. Then you can test it on the LFW dataset with the following command:
    ```Shell
    run_eval.bat
    ```

### Visualization of Uncertainty
TODO


## <img src="https://image.flaticon.com/icons/svg/48/48541.svg" width="25"/> Pre-trained Model

#### ResFace64
Base Mode: [Google Drive](https://drive.google.com/open?id=15CMJ4vh2_fdX3M05CIJY7V2B0ydoSr2Q)  
ProbFace: [Google Drive](https://drive.google.com/open?id=1R-sl5vaxWheyQBpgtONiSH5Qt4153Tl3)

#### ResFace64(0.5)
Base Mode: [Google Drive](https://drive.google.com/open?id=15CMJ4vh2_fdX3M05CIJY7V2B0ydoSr2Q)  
ProbFace: [Google Drive](https://drive.google.com/open?id=1R-sl5vaxWheyQBpgtONiSH5Qt4153Tl3)

Note: In the paper we used a different version of Ms-Celeb-1M. According to the authors of ArcFace, this dataset (MS-ArcFace) has already been cleaned and has no overlap with the test data.

#### Test Results: 
| Model | Method | LFW | IJB-A (FAR=0.1%) |
| ----- | ------ |--- | ----- |
| 64-CNN CASIA-WebFace | Baseline | 99.20 | 83.21 |
| 64-CNN CASIA-WebFace | PFE | 99.47 | 87.53 |
| 64-CNN Ms-ArcFace | Baseline | 99.72 | 91.93 |
| 64-CNN Ms-ArcFace | PFE | 99.83 | 94.82 |

(The PFE models and test results are obtained using exactly this demo code)

