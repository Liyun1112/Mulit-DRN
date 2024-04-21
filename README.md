MDRN: Multi-scale dual regression networks for single-image
super-resolution

Pytorch implementation for "MDRN: Multi-scale dual regression networks for single-image
super-resolution".


## Dependencies
```
Python>=3.7, PyTorch>=1.1, numpy, skimage, imageio, matplotlib, tqdm
```



## Quickstart (Model Testing)

You can evaluate our models on several widely used [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), including Set5, Set14, B100, Urban100, Manga109. Note that using an old PyTorch version (earlier than 1.1) would yield wrong results.

Please organize the benchmark datasets using the following hierarchy.
```
- data
    - benchmark
         - Set5
            - LR_bicubic
                - X4
                    - babyx4.png
```
You can use the following script to obtain the testing results:
```bash
python main.py --data_dir $DATA_DIR$ \
--save $SAVE_DIR$ --data_test $DATA_TEST$ \
--scale $SCALE$ --model $MODEL$ \
--pre_train $PRETRAINED_MODEL$ \
--test_only --save_results
```

- DATA_DIR: path to save data
- SAVE_DIR: path to save experiment results
- DATA_TEST: the data to be tested, such as Set5, Set14, B100, Urban100, and Manga109
- SCALE: super resolution scale, such as 4 and 8
- MODEL: model type, such as MDRN
- PRETRAINED_MODEL: path of the pretrained model


For example, you can use the following command to test our MDRN model for 4x SR.

```bash
python main.py --data_dir ~/data \
--save ../experiments --data_test Set5 \
--scale 4 --model MDRN \
--pre_train ../pretrained_models/MDRNS4x.pt \
--test_only --save_results
```

If you want to load the pretrained dual model, you can add the following option into the command.

```
--pre_train_dual ../pretrained_models/MDRN4x_dual_model.pt
```



## Training Method

We use DF2K dataset (the combination of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets) to train DRN-S and DRN-L.

```bash
python main.py --data_dir $DATA_DIR$ \
--scale $SCALE$ --model $MODEL$ \
--save $SAVE_DIR$
```

- DATA_DIR: path to save data
- SCALE: super resolution scale, such as 4 and 8
- MODEL: model type, such as MDRN
- SAVE_DIR: path to save experiment results

```

```

