import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class SHAPGradientExplainer:
    def __init__(self, model: tf.keras.Model, background_data: np.ndarray, n_samples: int = 50):
        """
        Initialize the SHAP Gradient Explainer.

        Parameters:
        - model: The TensorFlow/Keras model to explain.
        - background_data: Background dataset to integrate over (numpy array).
        - n_samples: Number of interpolation steps between the background and input sample.
        """
        self.model = model
        self.background_data = background_data
        self.n_samples = n_samples

    def explain(self, inputs: np.ndarray, target: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for the inputs.

        Parameters:
        - inputs: Input data to explain (numpy array of shape [batch_size, ...]).
        - target: The target class index for which gradients are computed.

        Returns:
        - shap_values: SHAP values for the inputs.
        """
        try:
            # Ensure inputs and background_data are numpy arrays
            inputs = np.array(inputs)
            background_data = np.array(self.background_data)

            # Initialize the shap values array
            shap_values = np.zeros_like(inputs, dtype=np.float32)

            # Number of background samples
            n_background = background_data.shape[0]

            # Loop over each input sample
            for i in range(inputs.shape[0]):
                input_sample = inputs[i:i+1]
                # Initialize an array to store attributions for each background sample
                attributions = np.zeros_like(input_sample, dtype=np.float32)

                # Loop over background samples
                for bg_sample in background_data:
                    bg_sample = bg_sample.reshape(input_sample.shape)
                    # Generate alphas for interpolation
                    alphas = np.linspace(0, 1, self.n_samples)
                    # Initialize a list to collect gradients
                    gradient_samples = []

                    for alpha in alphas:
                        # Interpolate between background and input
                        interpolated_input = bg_sample + alpha * (input_sample - bg_sample)
                        with tf.GradientTape() as tape:
                            tape.watch(interpolated_input)
                            prediction = self.model(interpolated_input)
                            if target is not None:
                                prediction = prediction[:, target]
                            else:
                                prediction = prediction[:, 0]
                        gradients = tape.gradient(prediction, interpolated_input)
                        gradient_samples.append(gradients.numpy())

                    # Average gradients over interpolation steps
                    avg_gradients = np.mean(np.array(gradient_samples), axis=0)
                    # Compute attributions for this background sample
                    attributions += (input_sample - bg_sample) * avg_gradients

                # Average attributions over all background samples
                shap_values[i] = attributions[0] / n_background  # Remove batch dimension

            return shap_values
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            return None

    def visualize(
        self,
        shap_values: np.ndarray,
        original_image: np.ndarray,
        cmap: str = 'viridis',
        alpha: float = 0.5,
        title: str = 'SHAP Values',
        show_colorbar: bool = True,
        figsize: tuple = (8, 8)
    ):
        """
        Visualize SHAP values for an image.

        Parameters:
        - shap_values: SHAP values to visualize (numpy array).
        - original_image: Original input image (numpy array).
        - cmap: Colormap for visualization.
        - alpha: Transparency level for overlay.
        - title: Title for the plot.
        - show_colorbar: Whether to display the colorbar.
        - figsize: Size of the figure.
        """
        # Aggregate SHAP values over color channels if necessary
        if shap_values.ndim == 3 and shap_values.shape[2] > 1:
            shap_values = np.mean(shap_values, axis=2)

        # Normalize SHAP values for visualization
        max_val = np.max(np.abs(shap_values))
        if max_val > 0:
            shap_values_norm = shap_values / max_val
        else:
            shap_values_norm = shap_values

        plt.figure(figsize=figsize)
        plt.imshow(original_image.astype(np.uint8))
        plt.imshow(shap_values_norm, cmap=cmap, alpha=alpha)
        if show_colorbar:
            plt.colorbar()
        plt.title(title)
        plt.axis('off')
        plt.show()
