# Neural Wireframe Renderer: Learning Wireframe to Image Translations
Pytorch implementation of ideas from the paper [Neural Wireframe Renderer: Learning Wireframe to Image Translations](https://arxiv.org/pdf/1912.03840.pdf) by Yuan Xue, Zihan Zhou, and Xiaolei Huang

### Dependencies
* Tested on CentOS 7
* Python >= 3.6
* [PyTorch](https://pytorch.org) >= 1.0
* TensorboardX >= 1.6

### Dataset
* You can download the data from [here](https://github.com/huangkuns/wireframe). By default, pelease extract all files inside ``v1.1`` to the ``data/raw_data/imgs`` folder, and extract all files inside ``pointlines``  to the ``data/raw_data/pointlines`` folder.
* To preprocess the data, run
```
python data/preprocess.py --uni_wf
```
The processed data will be saved under the ``data`` folder.


### Train
We support both single gpu training and multi-gpu training with Jiayuan Mao's [Synchronized Batch Normalization](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).

**Example Single GPU Training**

If you are training with color guided rendering:
```
python train.py --gpu 0 --batch_size 14
```
If you are training without color guided rendering:
```
python train.py --gpu 0 --batch_size 14 --nocolor
```

**Example Multiple GPU Training**
```
python train.py --gpu 0,1,2,3 --batch_size 40
```

**Tensorboard Visualization**
```
tensorboard --logdir results/tb_logs/wfrenderer --port 6666
```

### Test 
Note that the --nocolor option needs to be used consistently with training. For instance, you cannot train with --nocolor and test without --nocolor.
```
python test.py --gpu 0 --model_path YOUR_SAVED_MODEL_PATH --out_path YOUR_OUTPUT_PATH
```


### Input Modality
For now we only support rasterized wireframes as input, we will release the vectorized wireframe version in the near future.


### Citation
We hope our implementation can serve as a baseline for wireframe rendering. If you find our work useful in your research, please consider citing:
```
@inproceedings{xue2020neural,
  title={Neural Wireframe Renderer: Learning Wireframe to Image Translations},
  author={Xue, Yuan and Zhou, Zihan and Huang, Xiaolei},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

### Acknowledgement
Part of our code is adapted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
We also thank these great repos utilized in our code: [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [MSSSIM](https://github.com/jorge-pessoa/pytorch-msssim), [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch), 
