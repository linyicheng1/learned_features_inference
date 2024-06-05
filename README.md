# learned_features_inference
Implementing c++ inference for various learning point features

## inference time
| SuperPoint | ALIKE-T   | D2Net      | DISK         | XFeat        |
|------------|-----------|------------|--------------|--------------|
| 28.166 ms  | 70.038 ms | 109.992 ms | 408.688 ms | 73.582 ms |

## Examples 

- SuperPoint
![SuperPoint](images/sp.jpg "SuperPoint")

- ALIKE-T
![ALIKE-T](images/alike.jpg "ALIKE-T")
- DISK
![DISK](images/disk.jpg "DISK")
- D2Net
![D2Net](images/d2net.jpg "D2Net")
- XFeat
![XFeat](images/xfeat.jpg "XFeat")


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
```

```bash
./openvino_demo alike ../weight/Alike.xml ../images/1.jpg ../images/2.jpg
```
