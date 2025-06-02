# OCINet
This project provides the code and results for 'Ordered Cross-Scale Interaction Network for No-Service Rail Surface Defect Segmentation', IEEE TIM, 2025. [IEEE](https://ieeexplore.ieee.org/document/11018271) [Homepage](https://mathlee.github.io/)

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/OCINet/blob/main/images/OCINet.png">
   </div>

# Requirements
   python 3.8 + pytorch 1.9.0

# Saliency maps
   We provide [segmentation maps](https://pan.baidu.com/s/1oXRsAWJLpat-RydXWlfvTQ) (code: gaw6) of our OCINet and 19 compared methods on the NRSD-MN dataset.
      
   ![Image](https://github.com/MathLee/OCINet/blob/main/images/table.png)

# Training   
   Download [pvt_v2_b2.pth](https://pan.baidu.com/s/1U6Bsyhu0ynXckU6EnJM35w) (code: sxiq), and put it in './model/'. 

   Download [dataset.pth]() (code: ), and unzip it. Then, we use data_aug.m for data augmentation.
   
   Modify paths of datasets, then run train_OCINet.py.

Note: Our main model is under './model/OCINet_models.py'



# Pre-trained model and testing
1. Download the [pre-trained model](https://pan.baidu.com/s/1UHg0bOiPRTJwWzpu89Eyog) (code: hryz) on NRSD-MN dataset, and put it in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test_OCINet.py.


# Evaluation Tool
   You can use 'evaluation_Dice.py' to evaluate the segmentation maps.
   
# Our related work
[2025_IEEE SJ_DiffRSDI](https://github.com/zeroyi37/DiffRSDI)

[2023_IEEE TIM_NaDiNet](https://github.com/monxxcn/NaDiNet)

[2022_IEEE TIM_TSERNet](https://github.com/monxxcn/TSERNet)

[2022_Measurement_CSEPNet](https://github.com/showmaker369/CSEPNet)


# Citation
        @ARTICLE{Li_2025_OCINet,
                author = {Gongyang Li and Xiaofei Zhou and Hongyun Li},
                title = {Ordered Cross-Scale Interaction Network for No-Service Rail Surface Defect Segmentation},
                journal = {IEEE Transactions on Instrumentation and Measurement},
                volume = {},
                pages = {},
                year = {2025},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please get in touch with me at lllmiemie@163.com or ligongyang@shu.edu.cn.

