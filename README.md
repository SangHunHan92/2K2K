# High-fidelity 3D Human Digitization from Single 2K Resolution Images (2K2K)
This repository contains the code of the 2K2K method for 3D human reconstruction.

Sang-Hun Han, 
Min-Gyu Park, 
Ju Hong Yoon, 
Ju-Mi Kang, 
Young-Jae Park, and 
<a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ">Hae-Gon Jeon</a>
<br>Accepted to 
<a href="https://cvpr2023.thecvf.com/">CVPR 2023</a>

<a href="https://arxiv.org/abs/2303.15108">Paper</a> | 
<a href="https://sanghunhan92.github.io/conference/2K2K/">Project Page</a> | 
<a href="https://github.com/ketiVision/2K2K">Dataset</a>

<!-- [![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2303.15108) -->

<br/>
<p align="center">
  <img src="https://github.com/SangHunHan92/SangHunHan92.github.io/blob/main/imgs/2K2K/hanni_zoom_resize2.gif?raw=true" alt="Sublime's custom image"/>
</p>

<!-- ![Teaser Image](https://github.com/SangHunHan92/SangHunHan92.github.io/blob/main/imgs/2K2K/hanni_zoom_resize2.gif?raw=true) -->


## 2K2K Method
<br/>
<div align='center'><img src="https://github.com/SangHunHan92/SangHunHan92.github.io/blob/main/imgs/2K2K/Figure_4.png?raw=true" width=100%></div>

* Part-wise Normal Prediction divides an image into each body part through a matrix using keypoints. <br> This helps in predicting detailed normal vectors for each body part with minimal computation.
* Coarse-to-Fine Depth Prediction predicts a high-resolution depth map with very few network parameters and minimal memory usage.

## Installation
### Environment
* All models and codes work on Ubuntu 20.04 with Python 3.8, PyTorch 1.9.1 and CUDA 11.1.
You can install it by choosing one of the two methods below.

### Ubuntu Installation
* Install libraries with the following commands:
```bash
apt-get install -y freeglut3-dev libglib2.0-0 libsm6 libxrender1 libxext6 openexr libopenexr-dev libjpeg-dev zlib1g-dev
apt install -y libgl1-mesa-dri libegl1-mesa libgbm1 libgl1-mesa-glx libglib2.0-0
pip install -r requirements.txt
```

### Docker Installation
1. Create Docker Image From Dockerfile
```bash
docker build -t 2k2k:1.0 .
```

2. Make Docker Container From Image (example below)
```bash
docker run -e NVIDIA_VISIBLE_DEVICES=all -i -t -d --runtime=nvidia --shm-size=512gb --name 2k2k --mount type=bind,source={path/to/2k2k_code},target=/workspace/code 2k2k:1.0 /bin/bash
```

## Dataset Preparing
* We used RenderPeople, THuman2.0, and 2K2K datasets for training. A structure of the dataset folder will be formed as follows:
```bash
data
├ IndoorCVPR09
│  ├ airport_inside
│  └ ⋮
├ Joint3D
│  ├ RP
│  └ THuman2
├ list
├ obj
│  ├ RP
│  │  ├ rp_aaron_posed_013_OBJ
│  │  └ ⋮
│  └ THuman2
│     ├ data
│     └ smplx
├ PERS
│  ├ COLOR
│  ├ DEPTH
│  └ keypoint
└ (ORTH)
```

### Background Images
* You can download Background images from <a href="https://web.mit.edu/torralba/www/indoor.html">Recognizing Indoor Scenes</a>. Unzip this files under `data/IndoorCVPR09/`.


### Render Dataset (Image, Depth)
* To render the human datasets into images and depth maps, download the mesh files to `data/obj` first. See the folder structure above for download locations.
* For RenderPeople dataset, enter the command below to render. This will create folders `PERS` and `ORTH` (Optional). It takes about 2-3 days to render a 2048×2048 resolution images.
```bash
python render/render.py --data_path ./data --data_name RP
```
* For THuman2.0 dataset, you should use the SMPL-X model to render the front of human scans. Please download <a href="https://smpl-x.is.tue.mpg.de/download.php">SMPL-X</a> models anywhere. The `smplx` folder should exist under the `{smpl_model_path}`.

* After download SMPL-X models, you can render images.
```bash
python render/render.py --data_path ./data --data_name THuman2 --smpl_model_path {smpl_model_path}
```
### Render Dataset (Keypoint)
* Unzip 3D keypoints of RenderPeople and THuman2.0 dataset `Joint3D.zip` under `data/Joint3D`.
```bash
unzip data/Joint3D.zip -d data/Joint3D/
```
<!-- * Download 3D keypoints of RenderPeople and THuman2.0 dataset to `{data_folder}/Joint3D` from `!!!!!!` -->

* For RenderPeople training dataset, enter the command below to get 2D keypoints.
```bash
python render/render_keypoint.py --data_path ./data --data_name RP
```
* For THuman2.0 training dataset, enter the command below to get 2D keypoints.
```bash
python render/render_keypoint.py --data_path ./data --data_name THuman2
```
<!-- **Baseline models** -->

## Model Training
* Our model is divided into phase 1 and phase 2, learning high-resolution normal and depth respectively. To train phase 1, type follows,
```bash
python train.py --data_path ./data --phase 1 --batch_size 1
```
* After training phase 1, use pre-trained checkpoints to train phase 2, 
```bash
python train.py --data_path ./data --phase 2 --batch_size 1 --load_ckpt {checkpoint_file_name}
```
* If you want to train model with Distributed Data Parallel(DDP), use following code. You can also change options of `argparse` in `train.py` manually.
```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    train.py --use_ddp=True
```

## Model Test
* For test our model, we use <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">openpose</a> to extract 2d keypoints. We used <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md">Windows Portable Demo</a> for get `json` file.
* Put the image in a folder and run openpose like the code below will create a `json` keypoint file. Please refer `./test` folder. 
```bash
bin\OpenPoseDemo.exe --image_dir {test_folder} --write_json {test_folder} --hand --write_images {test_folder}\test --write_images_format jpg
```

* <del>Download the checkpoint file for quick results.</del> 
* Temporarily remove checkpoints due to licensing issue.
```bash
cd checkpoints && wget https://github.com/SangHunHan92/2K2K/releases/download/Checkpoint/ckpt_bg_mask.pth.tar && cd ..
```

* You can inference our model easily. This results depth map, normal map, and depth pointclouds `ply`.
```bash
python test_02_model.py --load_ckpt {checkpoint_file_name} --save_path {result_save_folder}
```

* To run the Poisson surface reconstruction, run the code below. Depending on your CPU performance, it will take between 1 and 10 minutes per object.
```bash
python test_03_model.py --save_path {result_save_folder}
```

## 2K2K Dataset
* Consisting of 2,050 3D human models from 80 DSLR cameras.
* Due to watermarking, the dataset will be released on June 16th.

## Citation
```bash
@inproceedings{han2023high,
  title={High-fidelity 3D Human Digitization from Single 2K Resolution Images},
  author={Han, Sang-Hun and Park, Min-Gyu and Yoon, Ju Hong and Kang, Ju-Mi and Park, Young-Jae and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
