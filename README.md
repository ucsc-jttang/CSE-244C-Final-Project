# 244C Medical Segmentation Project
This repo holds code from [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf) and DA-TransUNet: Integrating Positional and Channel Dual Attention with Transformer-Based U-Net for Enhanced Medical Image Segmentation (https://arxiv.org/abs/2310.12570)

## Usage

### 2. Obtain backbone, training, and test data

Download the data and models from: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd and recreate the same file structure so that the repo root, models, and data are on the same level as in below
![image](https://github.com/ucsc-jttang/CSE-244C-Final-Project/assets/160563338/08201b5d-53ec-47de-8d19-0f7bd865b968)


### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- To train, specify the architecture, else it will default to TransUnet. In the example outputs, a batch size of 20 was used due to memory constraints with the DATUnet model. 

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 20 --architecture DATU 

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 20 --architecture TU

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 20 --architecture CATU

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 20 --architecture PATU
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes. Specify the architecture to be tested or else it defaults to TransUnet

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16 --architecture PATU --batch_size 20

python test.py --dataset Synapse --vit_name R50-ViT-B_16 --architecture CATU --batch_size 20

python test.py --dataset Synapse --vit_name R50-ViT-B_16 --architecture DATU --batch_size 20

python test.py --dataset Synapse --vit_name R50-ViT-B_16 --batch_size 20

```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}

@article{sun2023transunet, title={DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation}, author={Sun, Guanqun and Pan, Yizhi and Kong, Weikun and Xu, Zichang and Ma, Jianhua and Racharak, Teeradaj, Nguyen, Le-Minh, Junyi Xin}, journal={arXiv preprint arXiv:2310.12570}, year={2023} }
```
