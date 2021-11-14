# CuteCNN：A Tiny Convolution Neural Network Works on Cifar10

This repository presents a tiny convolution neural network to accomplish the assignments of Artificial Intelligence Security 2021-FALL. The CuteCNN is trainable and can be evaluated on CIFAR10. The Super Parameters are set up artificially when running the code so that you can compare the  outcomes under  different conditions.

## CIFAR10

The CIFAR-10 dataset consists of 60000 32x32 color images in 10  classes, with 6000 images per class. There are 50000 training images and 10000 test images. The 10 classes are __Airplane__, __Automobile__, __Bird__, __Cat__, __Deer__, __Dog__, __Frog__, __Horse__,  __Ship__, __Truck__, as shown in the figure below.

![](https://github.com/EvaCharon/CuteCNN/blob/master/Picture/Readme/CIFAR10.jpg)

## CuteCNN - Modal Structure

The CuteCNN contains 4 convolution layers attached with max pooling layer, using 2 dense layers to output the 10-dim score vector , shown as below:

![](https://github.com/EvaCharon/CuteCNN/blob/master/Picture/Readme/Layers.jpg)

The input shape of the CuteCNN is 32×32 corresponding to the size of image in CIFAR10 dataset. To predict the class of external picture, some pretreatment will be processed to reshape the image. 

***

#### usage

The code should run on Jupyter Notebook  and the result can clearly display. The super parameters can be modified before the main method in the last code box, shown as below:

```python
#Loss function.
#"mean_squared_error"-"MSE"  
#"mean_absolute_error"-"MAE"
#"mean_absolute_percentage_error"-"MAPE"
# more in "https://keras.io/zh/losses/"
LOSS = 'MSE' 				

#Optimizer. choose from "adam","SGD","Adagrad"
#more in "https://keras.io/zh/optimizers/"
OPTIMIZER = 'adam'

#Activation function. Choose from"relu","tanh","sigmoid"
#more in "https://keras.io/zh/activations/"
ACTIVATION = 'relu'			

#The proportion of training data used as the validation set.
VALIDATION_SPLIT=0.2
#The batch size
BATCH_SIZE=100
#The epochs during training process
EPOCHS=100	

#The path to store the model. Can be None.
save_model_path=""

#The path to load existing weights. 
#If be None, then train the model from beginning
load_model_path=""			

#The image path that you willing to predict
img_path=""					
```

After setting these parameters, merely run the main method and you will get the results.

---

### Comparing the performance when tuning the super parameter

![](https://github.com/EvaCharon/CuteCNN/blob/master/Picture/Readme/activate.jpg)

epochs = 100 and batch_size=100:

As shown in the picture above, when using the different activation function Sigmoid brings about the best accuracy after 100 epochs. As the three methods didn't perform badly different from each other, we can tell that the cuteCNN met its bottleneck during training. When the relu function got the lowest train loss but  performed worst, it shows that the model may fall into over fitting. 

 ![](https://github.com/EvaCharon/CuteCNN/blob/master/Picture/Readme/optimizer.jpg)



epochs = 100 and batch_size=100, using MSE and Relu

When choosing different optimizer, the outcomes differs a lot. The training uses the default parameters of the optimizer, like learning rate. The results shows that model didn't work under the SGD and Adagrad, whose loss reduce too slowly to fit. Maybe it would be better if different learning rate was chosen.





