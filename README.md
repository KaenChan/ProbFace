# ProbFace


This is a demo code of training and testing [ProbFace] using Tensorflow.
ProbFace is a reliable Probabilistic Face Embeddging (PFE) method.
The representation of each face will be an Guassian distribution parametrized by (mu, sigma), where mu is the original embedding and sigma is the learned uncertainty. Experiments show that ProbFace could
+ improve the robustness of PFE.
+ simplify the calculation of the multal likelihood score (MLS).
+ improve the recognition performance on the risk-controlled scenarios.


## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Usage

### Preprocessing

Download the MS-Celeb-1M dataset from [insightface](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) or [baidu]() and decode it using [this code](https://github.com/deepinsight/insightface/blob/master/recognition/data/rec2image.py)

### Training
1. Download the base model [ResFace64](https://drive.baidu.com/open?id=1MiC_qCj5GFidWLtON9ekClOCJu6dPHT4) and unzip the files under ```log/resface64```.

2. Modify the configuration files under ```configfig/``` folder.

4. Start the training:
    ``` Shell
    python train.py configfig/resface64_msarcface.py
    ```
   
    ```
    Start Training
    name: resface64
    # epochs: 12
    epoch_size: 1000
    batch_size: 128

    Saving variables...
    Saving metagraph...
    Saving variables...
    [1][1] time: 4.19 a 0.8130 att_neg 2.7123 att_pos 0.9874 atte 1.8354 lr 0.0100 mls 0.6820 regu 0.1267 s_L2 0.0025 s_max 0.4467 s_min 0.2813
    [1][101] time: 37.72 a 0.8273 att_neg 2.9455 att_pos 1.0839 atte 1.8704 lr 0.0100 mls 0.6946 regu 0.1256 s_L2 0.0053 s_max 0.4935 s_min 0.2476
    [1][201] time: 38.06 a 0.8533 att_neg 2.9560 att_pos 1.1092 atte 1.9117 lr 0.0100 mls 0.7208 regu 0.1243 s_L2 0.0063 s_max 0.5041 s_min 0.2505
    [1][301] time: 38.82 a 0.7510 att_neg 2.9985 att_pos 1.0223 atte 1.7441 lr 0.0100 mls 0.6209 regu 0.1231 s_L2 0.0053 s_max 0.4552 s_min 0.2251
    [1][401] time: 37.95 a 0.8122 att_neg 2.9846 att_pos 1.0803 atte 1.8501 lr 0.0100 mls 0.6814 regu 0.1219 s_L2 0.0070 s_max 0.4964 s_min 0.2321
    [1][501] time: 38.42 a 0.7307 att_neg 3.0087 att_pos 1.0050 atte 1.8465 lr 0.0100 mls 0.6005 regu 0.1207 s_L2 0.0076 s_max 0.5249 s_min 0.2181
    [1][601] time: 37.69 a 0.7827 att_neg 3.0395 att_pos 1.0703 atte 1.8236 lr 0.0100 mls 0.6552 regu 0.1195 s_L2 0.0062 s_max 0.4952 s_min 0.2211
    [1][701] time: 37.36 a 0.7410 att_neg 2.9971 att_pos 1.0180 atte 1.8086 lr 0.0100 mls 0.6140 regu 0.1183 s_L2 0.0068 s_max 0.4955 s_min 0.2383
    [1][801] time: 37.27 a 0.6889 att_neg 3.0273 att_pos 0.9755 atte 1.7376 lr 0.0100 mls 0.5635 regu 0.1171 s_L2 0.0065 s_max 0.4773 s_min 0.2481
    [1][901] time: 37.34 a 0.7609 att_neg 2.9962 att_pos 1.0403 atte 1.8056 lr 0.0100 mls 0.6367 regu 0.1160 s_L2 0.0064 s_max 0.4861 s_min 0.2272
    Saving variables...
    --- cfp_fp ---
    testing verification..
    (14000, 96, 96, 3)
    # of images: 14000 Current image: 13952 Elapsed time: 00:00:12
    save /_feature.pkl
    sigma_sq (14000, 1)
    sigma_sq (14000, 1)
    sigma_sq [0.19821654 0.25770819 0.29024169 0.35030219 0.40342696 0.44539295
     0.56343746] percentile [0, 10, 30, 50, 70, 90, 100]
    risk_factor 0.0 risk_threshold 0.5634374618530273 keep_idxes 7000 / 7000 Cosine score acc 0.980429 threshold 0.182809
    risk_factor 0.1 risk_threshold 0.46279847621917725 keep_idxes 6301 / 7000 Cosine score acc 0.983336 threshold 0.201020
    risk_factor 0.2 risk_threshold 0.4453900158405304 keep_idxes 5603 / 7000 Cosine score acc 0.985007 threshold 0.203516
    risk_factor 0.3 risk_threshold 0.4327596127986908 keep_idxes 4904 / 7000 Cosine score acc 0.986134 threshold 0.207834
    ```

### Testing
+ **Single Image Comparison**
    We use LFW dataset as an example for single image comparison. Make sure you have aligned LFW images using the previous commands. Then you can test it on the LFW dataset with the following command:
    ```Shell
    run_eval.bat
    ```

### Visualization of Uncertainty

<img src="https://github.com/KaenChan/ProbFace/blob/master/log/ms1m-examples-choice-probface.jpg" width="600px">


## <img src="https://image.flaticon.com/icons/svg/48/48541.svg" width="25"/> Pre-trained Model

#### ResFace64
Base Mode: [Baidu Drive](https://pan.baidu.com/s/1ACjDBxA0tWFXs70J4dDv2A) v800, [Google Drive]()  
MLS Only: [Baidu Drive](https://pan.baidu.com/s/128A5r0q_NMvuQMUCs3WdCg) 72tt, [Google Drive]()  
MLS+L1+Triplet: [Baidu Drive](https://pan.baidu.com/s/1B4EtWymXe-E2WT7f7YHifA) sx8a, [Google Drive]()  
ProbFace: [Baidu Drive](https://pan.baidu.com/s/134XMGLIMd3iKx_9wRUH_Rg) pr0m, [Google Drive]()  

#### ResFace64(0.5)
Base Mode: [Baidu Drive](https://pan.baidu.com/s/1XJr0ZMxOPczEh62t9rg6qg) zrkl, [Google Drive]()  
MLS Only: [Baidu Drive](https://pan.baidu.com/s/1l4gD64yN3h0WYtqHap-KJw) et0e, [Google Drive]()  
MLS+L1+Triplet: [Baidu Drive](https://pan.baidu.com/s/1GX4oQOgmoWovqm2N-JXzCQ) glmf, [Google Drive]()  
ProbFace: [Baidu Drive](https://pan.baidu.com/s/1r10dVUpgrr3pifvd1LYB-g) o4tn, [Google Drive]()  

#### Test Results: 
| Method | LFW | IJB-A (FAR=0.1%) |
| ------ |--- | ----- |
| Baseline |  |  |
| PFE |  |  |
| Baseline |  |  |
| PFE | |  |


#### Acknowledgement

This repo is inspired by [Probabilistic-Face-Embeddings](https://github.com/seasonSH/Probabilistic-Face-Embeddings)