# Protest Activity Detection and Perceived Violence Estimation from Social Media Images

Implementation of the model used in the paper [Protest Activity Detection and Perceived Violence Estimation from Social Media Images](https://arxiv.org/abs/1709.06204) by Donghyeon Won, Zachary C. Steinert-Threlkeld, Jungseock Joo.

### Contents
[Requirements](#requirements)   
[Usage](#usage)

### Requirements   
[Pytorch](http://pytorch.org/)   
[NumPy](http://www.numpy.org/)   
[pandas](https://pandas.pydata.org/)   
[scikit-learn](http://scikit-learn.org/)   

### UCLA Protest Image Dataset   
![](https://raw.githubusercontent.com/wondonghyeon/protest-detection-violence-estimation/master/files/1-d.png)
You will need to download our UCLA Protest Image Dataset to train the model. Please e-mail me if you want to download our dataset!

#### Dataset Statistics   
\# of images: 40,764   
\# of protest images: 11,659   
##### Binary Variables (Protest \& Visual Attributes)

|Fields       |Protest|Sign  |Photo|Fire |Law Enf.|Children|Group>20|Group>100|Flag |Night|Shout|
|-------------|-------|------|-----|-----|--------|--------|--------|---------|-----|-----|-----|
|\# of Images |11,659 |9,669 |428  |667  |792     |347     |8,510   |2,939    |970  |987  |548  |
|Positive Rate|0.286  |0.829 |0.037|0.057|0.068   |0.030   |0.730   |0.252    |0.083|0.085|0.047|
##### Violence   

|Mean |Median |STD  |
|-----|-------|-----|
|0.365|0.352  |0.144|

![](https://raw.githubusercontent.com/wondonghyeon/protest-detection-violence-estimation/master/files/violence_hist.png)

### Model
#### Architecture   
We fine-tuned ImageNet pretrained [ResNet50](https://arxiv.org/abs/1512.03385) to our data. You can download the our trained model from this [Dropbox link](https://www.dropbox.com/s/hak8bp8zw8q6zfg/protest-model.pth.tar?dl=0).  
##### Performance

### Usage   
#### Training   
#### Evaluation
