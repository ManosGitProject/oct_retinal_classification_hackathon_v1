OCT Retinal Disease Classification using Deep Learning

Overview:
This repository implements deep learning models for multi-class OCT retinal disease classification (CNV, DME, DRUSEN, NORMAL) using ResNet-18 and 
DenseNet-121 architectures. 

The framework includes:
1) Data augmentation
2) Hard-case refinement (Hard Mining)
3) Class-specific weighted sampling
4) Calibration evaluation (ECE)
5) Grad-CAM visualization
6) High-confidence misclassification analysis

Dataset:
The experiments are based on the public OCT dataset (Kermany et al.), containing four classes:
CNV, DME, DRUSEN and NORMAL [Each sample is a retinal OCT B-scan image.]

Project Structure:
└── intruction manual.md
└── Final_Models
      └── densenet_aug
      └── densenet_no_aug
      └── resnet_aug
      └── resnet_no_aug
└── Plots
      └── densenet_aug
      └── densenet_no_aug
      └── resnet_aug
      └── resnet_no_aug
└── 01_EDA.ipynb
└── 02a_Train_ResNet18_Base.ipynb
└── 02b_Train_ResNet18_Augmented.ipynb
└── 02c_Train_DenseNet18_Base.ipynb
└── 02d_Train_DenseNet18_Augmented.ipynb
└── 03_Evaluation_Test_Data.ipynb
└── 04_GradCAM.ipynb
└── utils.py

Data Augmentation:
Training transformations include:
Resize
Mild ColorJitter (brightness/contrast)
RandomAffine (small rotation/translation)
Horizontal flip
Mild Gaussian blur
Normalization (ImageNet statistics)
!! Validation uses deterministic transforms only.

Evaluation Metrics:
The following metrics are reported:
Accuracy
Precision
Recall
Macro F1-score
Expected Calibration Error (ECE)
High-confidence misclassification rate (>0.9)

Grad-CAM Visualization:
To generate Grad-CAM heatmaps:
Class activation overlays
Visualization of disease-relevant retinal regions

Calibration (Future Work):
Post-hoc calibration (Temperature Scaling) can be applied to reduce overconfidence without affecting accuracy.

Typical performance:
Accuracy ≈ 96%
Macro F1 ≈ 0.95
ECE ≈ 0.03
High-confidence errors < 1%
Hard-case refinement significantly reduces high-confidence misclassifications.

Experiments were conducted on:
NVIDIA A100 (batch size 256 and 512) 

License:
This project is for research and academic purposes.
