# Keras RetinaNet Logo Detection
Keras implementation of RetinaNet object detection on logo detection. Forked on https://github.com/fizyr/keras-retinanet. Original paper is [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

## Preparation

1) Clone this repository.
2) In the repository `keras-retinanet` execute `python setup.py install --user`.
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.2 is installed.
3) This repository requires the master branch of `keras-resnet` (run `pip install --user --upgrade git+https://github.com/broadinstitute/keras-resnet`).

## Training

1) Make sure to complete Preparation steps first.
2) Download FlickrLogos32 dataset from [here](http://www.multimedia-computing.de/flickrlogos/data/FlickrLogos-32_dataset_v2.zip).
   Extract `FlickrLogos-v2` folder.
3) Download Logos-32plus_v1.0.1.zip dataset from [here](https://goo.gl/FPtqxR).
   Extract `images` folder and `groundtruth.mat` file to `Logos32plus` folder.
4) In main repository run command `python prepare.py -f ./../FlickrLogos-v2/ -l ./../Flickr32plus/ -c ./csvpaths/classes.csv -t ./csvpaths/retina-train.csv -v ./csvpaths/retina-valid.csv -s ./csvpaths/retina-test.csv`.
   Make sure `-f` option is FlickrLogos dataset folder path, `-l` is Logos32plus dataset path.
   In csvpaths folder files `retina-valid.csv`, `retina-train.csv` and `retina-test.csv` should have appeared.
5) Now run `python train.py -n name_of_snapshot_folder -c ./csvpaths/classes.csv -t ./csvpaths/retina-train.csv -v ./csvpaths/retina-valid.csv`.
   Folder with weighs and metadata should appear in `snapshots` folder.
   
## Evaluating classification model

1) Make sure to complete Preparation steps first.
   Make sure to do 2-4 Training steps. 
2) Train your own model or download weights from [here]().
3) In this repository run command `python evaluate.py -w weights.h5 -c ./csvpaths/classes.csv -t ./csvpaths/retina-test.csv -o ./evalkit/classification.txt`.
   `-w` is path to weights and `-o` is output path.
   In `evalkit` folder `classification.txt` file should appear.
4) In `evalkit` folder run command `python fl_eval_classification.py --flickrlogos=..\..\FlickrLogos-v2 --classification="classification.txt"`.
   Make sure `--flickrlogos` option is path to FlickLogos32 dataset and `--classification` option is txt file from step 3.
   You can use `original-classification.txt` which is made on default weights.

## Evaluating single photo or video

1) Make sure to complete Preparation steps first.
2) Train your own model or download weights from [here]().
3) To evaluate photo run command `python test.py -f ./examples/test.png -o ./examples/output.png -w weights.h5 -c ./csvpaths/classes.csv`.
   Where `-f` is your photo, `-o` is output photo, `-w` is weights.
4) To evaluate video run command `python test_video.py -f ./examples/video.mp4 -o ./examples/output_video.mp4 -w weights.h5 -c ./csvpaths/classes.csv`.

## Dependencies

Tensorflow (https://www.tensorflow.org/install/).
Keras 2.1.2 `pip install keras` install after Tensorflow.
OpenCV `pip install opencv-python`.