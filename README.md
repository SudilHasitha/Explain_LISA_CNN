# Explain LISA
It takes the following.
img: local path of img to be explained
class_names: the classes available as predictions for the given model
img_shape: shape of the image accepts by the neural network
model: the model to be explained get from tf.keras.models.load_model("your model path")
img1: local path background data point for produce explanations with SHAP
img2: local path background data point for produce explanations with SHAP
scale: for manual image scaling if scaling layer absent in the model to be explained 
filter_radius: the pixel value of the radius of the High pass filter

## Installation
```pip install Explain_LISA_CNN_Prediction```

## How to use it?
Open terminal and type python/python3 according to your OS.


``` import Explain_LISA_CNN_01 as e```
``` e.ExplainLISA(img,class_names,img_shape,model,img1,img2,scale,filter_radius)```

## License

© 2021 Sudil H.P Abeyagunasekera

This repository is licensed under the MIT license. See LICENSE for details.
