# JSPNet: Learning Joint Semantic & Instance Segmentation of Point Clouds via Similarity and Probability(PR under review)

## Overview
![](misc/fig.png)

## Dependencies

The code has been tested with Python 3.5 on Ubuntu 16.04.
*  [TensorFlow 1.4](https://www.tensorflow.org/)
*  h5py



## Data and Model

* Download 3D indoor parsing dataset (S3DIS Dataset). Version 1.2 of the dataset is used in this work.

``` bash
python utils/s3dis_utils/collect_indoor3d_data.py
python utils/s3dis_utils/s3dis_gen_h5.py
cd data && python generate_input_list.py && python generate_train_test_list.py
cd ..
```

* (optional) Prepared HDF5 data for training is available [here](https://drive.google.com/open?id=1PjWweT61nmIX7zc2vJClhzHxyTHGvstQ).



## Results

| baseline | S&I module | mIoU | mPre |                             para                             |
| :------: | :--------: | :--: | :--: | :----------------------------------------------------------: |
|  JSNet   |    w/o     | 52.3 | 52.0 |                                                              |
|  JSNet   |    SIFF    | 54.7 | 57.9 |                                                              |
|  JSNet   |    PIFF    | 55.2 | 60.2 | [model](https://drive.google.com/drive/folders/1rFhkmBHmNHfSMyUwRyHmKDriW_bl43Yx) |
|  JSNet   | SIFF&PIFF  | 55.8 | 59.3 | [model](https://drive.google.com/drive/folders/18W8xSoJ4a57KgdsFvhozrnJ3QQkxbxF3) |



## Usage

* Compile TF Operators

  Refer to [PointNet++](https://github.com/charlesq34/pointnet2)

* Training, Test, and Evaluation
``` bash
cd models/JSPNet/
ln -s ../../data .

# training
python train.py \
--gpu 0 \
--data_root ./ \
--data_type numpy \
--max_epoch  100  \
--log_dir ../../logs/train_5 \
--input_list data/train_file_list_woArea5.txt

# estimate_mean_ins_size 
python estimate_mean_ins_size.py \
--data_root ./ \
--input_list data/train_hdf5_file_list_woArea5.txt \
--out_dir ../../logs/train_5

# test
python test.py \
--gpu 0 \
--data_root ./ \
--data_type hdf5 \
--bandwidth 0.6   \
--num_point 4096  \
--log_dir ../../logs/test_5 \
--model_path ../../logs/train_5/epoch_99.ckpt \
--input_list  data/test_hdf5_file_list_Area5.txt

# evaluation
python eval_iou_accuracy.py --log_dir ../../logs/test_5
```

Note: We test on Area5 and train on the rest folds in default. 6 fold CV can be conducted in a similar way.

## Citation
If our work is useful for your research, please consider citing:

	paper under review
	we will release our paper on arxiv soon


## Acknowledgements
This code largely benefits from following repositories:
[ASIS](https://github.com/WXinlong/ASIS),
[PointNet++](https://github.com/charlesq34/pointnet2),
[PointConv](https://github.com/DylanWusee/pointconv),
[SGPN](https://github.com/laughtervv/SGPN) 
[DiscLoss-tf](https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow)

[JSNet](https://github.com/dlinzhao/JSNet)

[Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
