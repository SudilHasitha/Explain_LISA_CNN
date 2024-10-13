import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy

class LIMEImageExplainer:
    def __init__(self, kernel_width=0.25, verbose=False, random_state=None):
        """
        Initialize the LIME Image Explainer.

        Parameters:
        - kernel_width: Width of the kernel for weighting perturbations.
        - verbose: If True, prints additional information.
        - random_state: Seed or random state for reproducibility.
        """
        self.kernel_width = kernel_width
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,), hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         segmentation_fn=None):
        """
        Generate explanations for a prediction.

        Parameters:
        - image: 3D numpy array representing the image (H, W, C).
        - classifier_fn: Function that takes a numpy array (batch of images) and returns prediction probabilities.
        - labels: Iterable of labels to explain.
        - hide_color: Color to use for masking superpixels (if None, average color is used).
        - top_labels: If not None, produce explanations for the top K labels with highest probabilities.
        - num_features: Maximum number of features (superpixels) to include in the explanation.
        - num_samples: Number of perturbed samples to generate.
        - segmentation_fn: Function to segment the image into superpixels.

        Returns:
        - An ImageExplanation object containing the explanation.
        """
        # Convert grayscale to RGB if necessary
        if len(image.shape) == 2:
            image = gray2rgb(image)

        # Default segmentation function using SLIC
        if segmentation_fn is None:
            segmentation_fn = lambda x: slic(x, n_segments=50, compactness=10, sigma=1)
        segments = segmentation_fn(image)

        # Prepare the fudged image for masking
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = np.mean(image[segments == x], axis=(0, 1))
        else:
            fudged_image[:] = hide_color

        # Generate perturbed samples
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
        data[0, :] = 1  # Ensure the original image is included

        # Generate perturbed images
        labels_array = []
        imgs = []
        for row in tqdm(data):
            temp = image.copy()
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape, dtype=bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)

        # Get predictions for perturbed images
        preds = classifier_fn(np.array(imgs))
        labels_array = preds

        # Compute distances between perturbed samples and original image
        distances = self._compute_distances(data)

        # Create an explanation object
        explanation = ImageExplanation(image, segments)

        # Determine labels to explain
        if top_labels:
            top = np.argsort(labels_array[0])[-top_labels:]
            explanation.top_labels = list(top)
            labels_to_explain = top
        else:
            labels_to_explain = labels

        # Fit local models for each label
        for label in labels_to_explain:
            # Compute sample weights
            weights = self._kernel(distances)
            # Fit a linear model
            model = Ridge(alpha=1, fit_intercept=True)
            model.fit(data, labels_array[:, label], sample_weight=weights)
            # Store results in the explanation object
            coeff = model.coef_
            intercept = model.intercept_
            explanation.intercept[label] = intercept
            explanation.local_exp[label] = list(zip(range(n_features), coeff))
            explanation.score = model.score(data, labels_array[:, label], sample_weight=weights)
            explanation.local_pred = model.predict(data[0].reshape(1, -1))

        return explanation

    def _compute_distances(self, data):
        """
        Compute distances between perturbed samples and the original instance.

        Parameters:
        - data: Binary matrix indicating which superpixels are active.

        Returns:
        - distances: Array of distances.
        """
        distances = pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()
        return distances

    def _kernel(self, distances):
        """
        Compute sample weights using an exponential kernel.

        Parameters:
        - distances: Array of distances.

        Returns:
        - weights: Array of weights.
        """
        return np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))


class ImageExplanation:
    def __init__(self, image, segments):
        """
        Initialize the Image Explanation.

        Parameters:
        - image: Original image.
        - segments: Segmented image (superpixels).
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None
        self.top_labels = None
        self.score = None

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """
        Get the image and mask highlighting the explanation.

        Parameters:
        - label: Label to explain.
        - positive_only: If True, only include features that contribute positively to the prediction.
        - negative_only: If True, only include features that contribute negatively.
        - hide_rest: If True, make the rest of the image gray.
        - num_features: Number of features to include.
        - min_weight: Minimum weight to include a feature.

        Returns:
        - (temp, mask): Tuple of the image with explanations and the mask.
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only and negative_only:
            raise ValueError("positive_only and negative_only cannot be True at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, dtype=bool)
        if hide_rest:
            temp = np.zeros_like(image)
        else:
            temp = image.copy()
        if positive_only:
            fs = [x[0] for x in exp if x[1] > min_weight][:num_features]
        elif negative_only:
            fs = [x[0] for x in exp if x[1] < -min_weight][:num_features]
        else:
            fs = [x[0] for x in exp if np.abs(x[1]) > min_weight][:num_features]
        for f in fs:
            temp[segments == f] = image[segments == f]
            mask[segments == f] = True
        return temp, mask
