import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, Tuple, Optional, Callable, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import logging

logger = logging.getLogger(__name__)

class IntegratedGradients:
    def __init__(self, model: tf.keras.Model, n_steps: int = 50, method: str = 'gausslegendre', internal_batch_size: int = 100):
        """
        Initialize the Integrated Gradients explainer.

        Parameters:
        - model: The TensorFlow/Keras model to explain.
        - n_steps: Number of steps for interpolation between the baseline and the input.
        - method: Method for integral approximation ('riemann', 'trapezoidal', 'gausslegendre').
        - internal_batch_size: Batch size for internal computations.
        """
        self.model = model
        self.n_steps = n_steps
        self.method = method
        self.internal_batch_size = internal_batch_size

    def explain(self, inputs: np.ndarray, baselines: Optional[np.ndarray] = None, target: Optional[int] = None) -> np.ndarray:
        """
        Compute the Integrated Gradients for the inputs.

        Parameters:
        - inputs: The input data to explain (numpy array of shape [batch_size, height, width, channels]).
        - baselines: The baseline input (numpy array). If None, uses zeros.
        - target: The target class index for which gradients are computed.

        Returns:
        - attributions: The attributions for each input feature.
        """
        if baselines is None:
            baselines = np.zeros_like(inputs)

        # Generate alphas for interpolation
        alphas = self._generate_alphas()

        # Initialize an empty list to collect gradients
        gradient_batches = []

        # Batch processing to handle large inputs
        for batch_start in range(0, len(inputs), self.internal_batch_size):
            batch_end = min(batch_start + self.internal_batch_size, len(inputs))
            input_batch = inputs[batch_start:batch_end]
            baseline_batch = baselines[batch_start:batch_end]

            # Compute attributions for the batch
            batch_attributions = self._compute_batch_attributions(input_batch, baseline_batch, alphas, target)
            gradient_batches.append(batch_attributions)

        # Concatenate all batches
        attributions = np.concatenate(gradient_batches, axis=0)

        return attributions

    def _generate_alphas(self) -> np.ndarray:
        """
        Generate interpolation coefficients (alphas) based on the selected method.

        Returns:
        - alphas: Array of interpolation coefficients.
        """
        if self.method == 'riemann':
            alphas = np.linspace(0, 1, self.n_steps)
        elif self.method == 'trapezoidal':
            alphas = np.linspace(0, 1, self.n_steps)
        elif self.method == 'gausslegendre':
            alphas, _ = np.polynomial.legendre.leggauss(self.n_steps)
            alphas = (alphas + 1) / 2  # Scale from [-1, 1] to [0, 1]
        else:
            raise ValueError(f"Unsupported integration method: {self.method}")
        return alphas

    def _compute_batch_attributions(self, inputs: np.ndarray, baselines: np.ndarray, alphas: np.ndarray, target: Optional[int]) -> np.ndarray:
        """
        Compute attributions for a batch of inputs.

        Parameters:
        - inputs: Batch of input data.
        - baselines: Corresponding baselines.
        - alphas: Interpolation coefficients.
        - target: Target class index.

        Returns:
        - batch_attributions: Attributions for the batch.
        """
        batch_size = inputs.shape[0]
        expanded_inputs = np.expand_dims(inputs, axis=1)
        expanded_baselines = np.expand_dims(baselines, axis=1)
        expanded_alphas = alphas.reshape((1, -1, 1, 1, 1))

        # Interpolate inputs
        interpolated_inputs = expanded_baselines + expanded_alphas * (expanded_inputs - expanded_baselines)
        interpolated_inputs = interpolated_inputs.reshape((-1,) + inputs.shape[1:])

        # Compute gradients
        with tf.GradientTape() as tape:
            interpolated_inputs_tf = tf.convert_to_tensor(interpolated_inputs)
            tape.watch(interpolated_inputs_tf)
            predictions = self.model(interpolated_inputs_tf)
            if target is not None:
                predictions = predictions[:, target]
            else:
                predictions = tf.reduce_sum(predictions, axis=1)
        gradients = tape.gradient(predictions, interpolated_inputs_tf)
        gradients = gradients.numpy().reshape((batch_size, len(alphas)) + inputs.shape[1:])

        # Average gradients
        avg_gradients = np.mean(gradients, axis=1)

        # Compute attributions
        attributions = (inputs - baselines) * avg_gradients

        return attributions

    def visualize_attributions(
        self,
        attributions: np.ndarray,
        original_image: Union[None, np.ndarray] = None,
        method: str = "heat_map",
        sign: str = "absolute_value",
        outlier_perc: Union[int, float] = 2,
        cmap: Union[None, str] = None,
        alpha_overlay: float = 0.5,
        show_colorbar: bool = False,
        title: Union[None, str] = None,
        fig_size: Tuple[int, int] = (6, 6),
    ) -> Tuple[Figure, Axes]:
        """
        Visualize attributions for a given image.

        Parameters:
        - attributions: Numpy array of attributions (height, width, channels).
        - original_image: Original image (height, width, channels). Required for methods other than 'heat_map'.
        - method: Visualization method ('heat_map', 'blended_heat_map', 'original_image', 'masked_image', 'alpha_scaling').
        - sign: Attribution sign to visualize ('positive', 'negative', 'absolute_value', 'all').
        - outlier_perc: Percentile for outlier clipping (default: 2).
        - cmap: Colormap for heatmap.
        - alpha_overlay: Transparency level for overlays.
        - show_colorbar: Whether to display the colorbar.
        - title: Title for the plot.
        - fig_size: Figure size.

        Returns:
        - fig: Matplotlib Figure object.
        - ax: Matplotlib Axes object.
        """

        # Internal helper functions
        def _prepare_image(image):
            image = image.astype(np.uint8)
            return image

        def _normalize_attributions(attr, sign, outlier_perc):
            attr = attr.copy()
            if sign == 'positive':
                attr = np.maximum(0, attr)
            elif sign == 'negative':
                attr = np.maximum(0, -attr)
            elif sign == 'absolute_value':
                attr = np.abs(attr)
            elif sign == 'all':
                pass  # Use all attributions
            else:
                raise ValueError("Invalid sign option.")

            # Flatten the array for percentile computation
            attr_flat = attr.flatten()

            # Clip outliers
            if outlier_perc > 0:
                lower_bound = np.percentile(attr_flat, outlier_perc)
                upper_bound = np.percentile(attr_flat, 100 - outlier_perc)
                attr = np.clip(attr, lower_bound, upper_bound)
            else:
                lower_bound = attr.min()
                upper_bound = attr.max()

            # Normalize to [0, 1]
            if upper_bound - lower_bound > 0:
                attr = (attr - lower_bound) / (upper_bound - lower_bound)
            else:
                attr = np.zeros_like(attr)

            return attr

        # Prepare the original image
        if original_image is not None:
            if np.max(original_image) <= 1.0:
                original_image = _prepare_image(original_image * 255)
            else:
                original_image = _prepare_image(original_image)
        else:
            if method != 'heat_map':
                raise ValueError(f"Original image must be provided for method '{method}'.")

        # Normalize attributions
        norm_attr = _normalize_attributions(attributions, sign, outlier_perc)

        # Set default colormap
        if cmap is None:
            if sign == 'positive':
                cmap = 'Greens'
            elif sign == 'negative':
                cmap = 'Reds'
            elif sign == 'absolute_value':
                cmap = 'Blues'
            else:  # sign == 'all'
                cmap = LinearSegmentedColormap.from_list('RdWhGn', ['red', 'white', 'green'])

        # Create figure and axis
        fig, ax = plt.subplots(figsize=fig_size)

        # Visualization
        if method == 'heat_map':
            heatmap = ax.imshow(norm_attr, cmap=cmap)
        elif method == 'blended_heat_map':
            ax.imshow(np.mean(original_image, axis=2), cmap='gray')
            heatmap = ax.imshow(norm_attr, cmap=cmap, alpha=alpha_overlay)
        elif method == 'original_image':
            ax.imshow(original_image)
        elif method == 'masked_image':
            if sign == 'all':
                raise ValueError("Cannot use 'all' sign with 'masked_image' method.")
            masked_image = original_image * np.expand_dims(norm_attr, 2)
            ax.imshow(_prepare_image(masked_image))
        elif method == 'alpha_scaling':
            if sign == 'all':
                raise ValueError("Cannot use 'all' sign with 'alpha_scaling' method.")
            alpha_channel = _prepare_image(norm_attr * 255)
            if original_image.shape[2] == 3:
                rgba_image = np.dstack((original_image, alpha_channel))
            else:
                rgba_image = np.concatenate([original_image, alpha_channel[:, :, np.newaxis]], axis=2)
            ax.imshow(rgba_image)
        else:
            raise ValueError(f"Invalid method '{method}'.")

        # Remove axis ticks
        ax.axis('off')

        # Add title
        if title:
            ax.set_title(title)

        # Show colorbar if applicable
        if show_colorbar and method in ['heat_map', 'blended_heat_map']:
            fig.colorbar(heatmap, ax=ax)

        plt.show()

        return fig, ax

    def reset_predictor(self, model: tf.keras.Model) -> None:
        """
        Reset the model used for predictions.

        Parameters:
        - model: New TensorFlow/Keras model.
        """
        self.model = model
