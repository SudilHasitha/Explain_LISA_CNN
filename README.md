# LISA CNN Explainer V3

LISA CNN Explainer V3 is a tool for explaining CNN predictions using multiple explainability methods.

## Installation

```
pip install LISA_CNN_ExplainerV3
```

## Usage

To use LISA CNN Explainer V3, follow these steps:

1. Import the package:

```python
from LISA_CNN_ExplainerV3 import ExplainLISA
```

2. Load your model:

```python
import tensorflow as tf
model = tf.keras.models.load_model("path/to/your/model.h5")
```

3. Create an ExplainLISA instance and run the explanation:

```python
explainer = ExplainLISA(
    img="path/to/image.jpg",
    pred_class="Unknown",
    img_shape=224,
    model=model,
    img1="path/to/background_image1.jpg",
    img2="path/to/background_image2.jpg",
    scale=True,
    filter_radius=10
)
explainer.explain()
```

Parameters:
- `img`: Path to the image to be explained
- `pred_class`: Predicted class (can be "Unknown" if not known)
- `img_shape`: Shape of the image accepted by the neural network
- `model`: The model to be explained
- `img1` and `img2`: Paths to background data points for SHAP explanations
- `scale`: Whether to scale the image (set to False if your model includes a scaling layer)
- `filter_radius`: Pixel value of the radius of the high-pass filter

4. The explanation results will be saved as image files in your current directory:
   - AnchorSegmentation.png
   - LimeExplanation.png
   - IGExplanation.png
   - SHAPExplanation.png
   - LISAExplanation.png

## Command-line Usage

You can also use LISA CNN Explainer V3 from the command line:

```
python -m LISA_CNN_ExplainerV3.ExplainLISA --img path/to/image.jpg --model path/to/model.h5 --img_shape 224 --filter_radius 10
```

## License

© 2021 Sudil H.P Abeyagunasekera

This repository is licensed under the MIT license. See LICENSE for details.
