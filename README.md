# Sentiment Analysis Using BERT

## Introduction
Using Base Bert (12 Layers) is used to achive 96% accuracy on Binary Sentiment analysis of Movie Review Dataset.<br>

## Files 
There are 2 main noteooks Data
- Data_preparation.ipynb
- predictions.ipynb


Another python file is **dataloader_classes_func.py** . This file contains the class and function definition of pytorch Dataset and DataLoader respectively.<br>
<br>
In __Data_preparation.ipynb__ I the train and validation Data is preprocessesd and saved as .pkl file; so that later it can be read without doing preprocessing steps again.<br>
<br>
In __predictions.ipynb__ the model is created nd trained and saved the validation accuracy is __96.6%__ achieved.<br>

## Training
The model is trained on AWS p2.Xlarge Notebook instance. The model is trained for 7 epoches with training and validation improving till the last epoch . Each epoch takes around 25 minutes to complete.

## Results
I have managed to achieve __96.6%__ validation accuracy with 7 epoches of training. There is no test set provided with this dataset (though the *test* folder is there but it is empty). I had already used the train dataset as train and validationset.
