### install

- `conda create -n torch python=3.8`
- install pytorch

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

- install `requirements.txt`

```bash
pip install -r requirements.txt
```


### 生成mono prior
这里的mono prior主要指法向normal和深度depth先验，提供两种选择，一个是使用`omnidata`提供

#### omnidata
```bash
cd preprocess/omnidata
```
- install
```bash
sh download.sh
```
- 生成`depth`先验
```bash
python extract_mono_cues_rectangle.py --input_dir <imgs_dir> --output_dir <outdir> --task depth
```
- 生成`normal`先验
```bash
python extract_mono_cues_rectangle.py --input_dir <imgs_dir> --output_dir <outdir> --task normal
```

#### depth-anything
[depth-anything](https://github.com/LiheYoung/Depth-Anything)是香港大学2024年CVPR的深度估计工作，其在150万张标注图像和6200多万张未标注图像上训练，估计精度和完整度超过MiDas，训练分辨率为518*518。

```bash
cd preprocess/depth_anything
python run.py --img-path <imgs_dir> --outdir <outdir>
```

或者：
```bash
cd preprocess/depth_anything
sh extract_mono_depth.sh <data_dir>
```
- 要求`data_dir`下有`rgb`目录，其将结果保存到`data_dir/mono_depth`下，子目录`res`为`.npy`，`vis`为深度可视化。

#### surface-normal-uncertainty
[surface-normal-uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty)是2021年ICCV的单目法线和不确定性估计工作，其优势在于提供了uncertainty

有`scannet`和`nyu`得到的两个预训练模型。

```bash
cd preprocess/surface-normal-uncertainty
python run.py -i <imgs_dir> -n <normal_outdir> -u <uncertainty_outdir> --architecture BN --pretrained scannet
```

或者：
```bash
cd preprocess/surface-normal-uncertainty
sh extract_normal_uncertainty.sh <data_dir>
```

### 生成图片的 segmentic mask
这部分主要是为了分割出图像中的texture-less区域，即墙壁、天花板和地板。

- install
```bash
cd preprocess/mask/maskdino/modeling/pixel_decoder/ops/
sh make.sh
```

提供两种语义分割方法：[Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything) ，[MaskDINO](https://github.com/IDEA-Research/MaskDINO)。
- Lang-SAM 是一种基于SAM(segment-anything)的分割方法，通过text prompt 进行分割，这里使用`ceiling.wall.floor`作为prompt 分割出天花板，墙面，地面等区域
- MaskDINO 是一种有监督的语义分割方法，在后续融合过程中，选择`ceiling`,`wall`,`floor`的区域作为mask。

`Lang-SAM` 方法
```python
cd preprocess/mask
python sam.py -i <imgs_dir> -o <output_dir>
```

`MaskDINO` 方法
```python
cd preprocess/mask
python maskdino.py --config maskdino/maskdino_R50_bs16_160k_steplr.yaml \
    --input /path/to/image/**/*.jpg \
    --output /path/to/output
```


### 生成图片的normal和uncertainty
```bash

### dataset

####

#### custom dataset
用colmap构造自定义数据

```bash
cd preprocess
python colmap/convert.py -s <source_path>
```
- `convert.py` is adapted from the official gaussian-splatting code. The sparse reconstruction results are saved in the `source_path/sparse/0` folder, and the undistorted images are saved in the `source_path/images` folder.