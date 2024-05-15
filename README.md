# learned_features_inference
Implementing c++ inference for various learning point features

## Supported Features:
- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [ALIKE-T](https://github.com/Shiaoming/ALIKE)
- [DISK](https://github.com/cvlab-epfl/disk)
- [D2Net](https://github.com/mihaidusmanu/d2-net)
- [XFeat](https://github.com/pfnet-research/xfeat)
- More TODO 

## Supported Libraries:
- [OpenVINO](https://docs.openvino.ai/2022.3/home.html)
- [NCNN](https://github.com/Tencent/ncnn) (TODO)
- [TensorRT](https://developer.nvidia.com/tensorrt) (TODO)

## Usage:
```bash
./openvino_demo <model_type> <model_path> <image_0_path> <image_0_path>
./openvino_demo alike ../weight/Alike.xml ../images/1.jpg ../images/2.jpg
```
