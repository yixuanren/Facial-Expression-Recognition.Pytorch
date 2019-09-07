# Facial-Expression-Recognition.Pytorch
A CNN based pytorch implementation on facial expression recognition (FER2013 and CK+), achieving 73.112% (state-of-the-art) in FER2013 and 94.64% in CK+ dataset

## Demos ##
![Image text](https://raw.githubusercontent.com/WuJie1010/Facial-Expression-Recognition.Pytorch/master/demo/1.png)
![Image text](https://raw.githubusercontent.com/WuJie1010/Facial-Expression-Recognition.Pytorch/master/demo/2.png)

## Dependencies ##
- Python 2.7
- Pytorch >=0.4.0
- h5py (Preprocessing)
- sklearn (plot confusion matrix)

## Visualize for a test image by a pre-trained model ##
- Firstly, download the pre-trained model from https://drive.google.com/open?id=1Oy_9YmpkSKX1Q8jkOhJbz3Mc7qjyISzU and then put it in the "FER2013_VGG19" folder; Next, Put the test image (rename as 1.jpg) into the "images" folder, then 
- `python visualize.py`

### Extract representations from other datasets ###
- There are two versions in terms of the representations by now: the initial version is the prediction confidence (the output of the last layer before softmax), whose dimension is 7; the second version the output of the second last layer (after activation), with the dimension of 512, which is the default option;
- Other datasets can be any ImageFolder dataset. For now I use a center-croped version of CelebA (128x128) shared at https://drive.google.com/file/d/1jsy1NN4LojSpjaG_BLvNJl-81TKbNaxj/view?usp=sharing;
- After downloading and unziping, you need to put the folder somewhere and modify the path in `visualize.py` correspondingly;
- Finally, run `python visualize.py`, and an `.npz` file will be dumped to disk with representations of all images in order. (The one I generated can be downloaded at https://drive.google.com/file/d/1GH7myZpmmLzA40i3G9p865l3TZLPWOAX/view?usp=sharing.)

## FER2013 Dataset ##
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

### Preprocessing Fer2013 ###
- first download the dataset(fer2013.csv) then put it in the "data" folder, then
- python preprocess_fer2013.py

### Train and Eval model ###
- python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest

###              fer2013 Accurary             ###

- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>
- Model：   Resnet18 ;     PublicTest_acc：  71.190% ;    PrivateTest_acc：72.973%     

## CK+ Dataset ##
- The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos,
We extracted the last three frames from each sequence in the CK+ dataset, which
contains a total of 981 facial expressions. we use 10-fold Cross validation in the experiment.

### Train and Eval model for a fold ###
- python mainpro_CK+.py --model VGG19 --bs 128 --lr 0.01 --fold 1

### Train and Eval model for all 10 fold ###
- python k_fold_train.py

### plot confusion matrix for all fold ###
- python plot_CK+_confusion_matrix.py --model VGG19

###      CK+ Accurary      ###
- Model：    VGG19 ;       Test_acc：   94.646%   <Br/>
- Model：   Resnet18 ;     Test_acc：   94.040%   

