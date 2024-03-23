# [LightM-UNet](https://arxiv.org/html/2403.05246v1)

Official repository for "LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation".

## Release


- 🔥 **News**: ```2024/3/17```: LightM-UNet released.

- **A regrettable notification**: ```2024/3/12``` Thank you for your attention! The author has been quite busy lately (；′⌒`), so only the main code has been uploaded to this repository for now. Detailed code explanations, data, and configuration details will be completed by March 17, 2024.
Thanks again for your interest! o(￣▽￣)ブ

## Introduction to LightM-UNet

LightM-UNet is a lightweight fusion of UNet and Mamba, boasting a mere parameter count of **1M**. Through validation on both 2D and 3D real-world datasets, LightM-UNet surpasses existing state-of-the-art models. In comparison to the renowned [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and contemporaneous [U-Mamba](https://github.com/bowang-lab/U-Mamba), LightM-UNet reduces the parameter count by **116X** and **224X**, respectively.

![result](https://github.com/MrBlankness/LightM-UNet/blob/master/assets/main_result.png)

## Get Start 

Requirements: `CUDA ≥ 11.6`

1. Create a virtual environment: `conda create -n lightmunet python=3.10 -y` and `conda activate lightmunet `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d==1.1.1` and `pip install mamba-ssm`
4. Download code: `git clone https://github.com/MrBlankness/LightM-UNet`
5. `cd LightM-UNet/lightm-unet` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Data Preparation

Download LiTs dataset [here](https://www.kaggle.com/datasets/gauravduttakiit/3d-liver-and-liver-tumor-segmentation) and Montgomery&Shenzhen dataset [here](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels), then put them into the `LightM-Unet/data/nnUNet_raw` folder. 
LightM-UNet is built on the popular [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. If you want to train LightM-UNet on your own dataset, please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset. 

Please organize the dataset as follows:

```
data/
├── nnUNet_raw/
│   ├── Dataset801_LiverCT/
│   │   ├── imagesTr
│   │   │   ├── Liver_0001_0000.nii.gz
│   │   │   ├── Liver_0002_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── Liver_0001.nii.gz
│   │   │   ├── Liver_0002.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── Dataset802_LungXray/
│   │   ├── imagesTr
│   │   │   ├── Lung_0001_0000.png
│   │   │   ├── Lung_0002_0000.png
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── Lung_0001.png
│   │   │   ├── Lung_0001.png
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── ...
```

Based on nnUNet, preprocess the data and generate the corresponding configuration files (the generated results can be found in the `LightM-Unet/data/nnUNet_preprocessed` folder).

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Model Training


### Train 2D models

- Train 2D `LightM-Unet` model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerLightMUNet
```

### Train 3D models

- Train 3D `LightM-Unet` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerLightMUNet
```


## Inference

### Inference 2D models

- Inference 2D `LightM-Unet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 2d -tr nnUNetTrainerLightMUNet --disable_tta
```

### Inference 3D models

- Inference 3D `LightM-Unet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -tr nnUNetTrainerLightMUNet --disable_tta
```


## Citation
If you find our work helpful, please consider citing the following papers
```
@misc{liao2024lightmunet,
      title={LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation}, 
      author={Weibin Liao and Yinghao Zhu and Xinyuan Wang and Chengwei Pan and Yasha Wang and Liantao Ma},
      year={2024},
      eprint={2403.05246},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```


## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. 
We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mrblankness/lightm-unet&type=Date)](https://star-history.com/#mrblankness/lightm-unet&Date)

