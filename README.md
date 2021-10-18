# CuteCNN：A Tiny Convolution Neural Network Works on Cifar10

This repository presents a tiny convolution neural network to accomplish the assignments of Artificial Intelligence Security 2021-FALL. The CuteCNN is trainable and can be evaluated on CIFAR10. The Super Parameters are set up artificially when running the code so that you can compare the  outcomes under  different conditions.

***

#### CIFAR10

The CIFAR-10 dataset consists of 60000 32x32 color images in 10  classes, with 6000 images per class. There are 50000 training images and 10000 test images. The 10 classes are __Airplane__, __Automobile__, __Bird__, __Cat__, __Deer__, __Dog__, __Frog__, __Horse__,  __Ship__, __Truck__, as shown in the figure below.

![](C:\Users\charo\Desktop\CIFAR10.jpg)

***



#### CuteCNN - Modal Structure

The CuteCNN contains 4 convolution layers attached with max pooling layer, using 2 dense layers to output the 10-dim score vector , shown as below:

![](C:\Users\charo\Desktop\Layers.jpg)

The input shape of the CuteCNN is 32×32 corresponding to the size of image in CIFAR10 dataset. To predict the class of external picture, some pretreatment will be processed to reshape the image. 

***

#### usage

The code should run on Jupyter Notebook  and the result can clearly display. The super parameters can be modified before the main method in the last code box, shown as below:

```python

LOSS = 'MSE' 				#Loss function.
							#"mean_squared_error"-"MSE"
							#"mean_absolute_error"-"MAE"
        					#"mean_absolute_percentage_error"-"MAPE"
							#"mean_squared_logarithmic_error" - "MAPE"
							# more in "https://keras.io/zh/losses/"

OPTIMIZER = 'adam'			#Optimizer. choose from "adam","SGD","Adagrad"
							#more in "https://keras.io/zh/optimizers/"

ACTIVATION = 'relu'			#Activation function. Choose from"relu","tanh","sigmoid"
							#more in "https://keras.io/zh/activations/"

 
VALIDATION_SPLIT=0.2		#The proportion of training data used as the validation set.
BATCH_SIZE=100				#The batch size
EPOCHS=100					#The epochs during training process

save_model_path=""			#The path to store the model. Can be None.

load_model_path=""			#The path to load existing weights. 
							#If be None, then train the model from beginning

img_path=""					#The image path that you willing to predict
```

After setting these parameters, merely run the main method and you will get the results.













