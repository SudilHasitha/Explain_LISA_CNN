import numpy as np
from skimage.segmentation import slic
from functools import partial
import copy

class Predictor:
    """
    A wrapper for your prediction function or model.
    """

    def __init__(self, predictor_fn, preprocessor=None):
        if not callable(predictor_fn):
            raise ValueError('Predictor function must be callable.')
        self.predictor_fn = predictor_fn
        self.preprocessor = preprocessor

    def __call__(self, x):
        if self.preprocessor:
            x = self.preprocessor.transform(x)
        return self.predictor_fn(x)


class ArgmaxTransformer:
    """
    Transforms model output probabilities into class labels.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, x):
        pred_probs = np.atleast_2d(self.predictor(x))
        return np.argmax(pred_probs, axis=1)


class AnchorImageExplainer:
    def __init__(self, predictor, image_shape, segmentation_fn='slic', segmentation_kwargs=None):
        """
        Initialize the AnchorImageExplainer.

        Parameters:
        - predictor: A callable that takes an array of images and returns predictions.
        - image_shape: The shape of the images (height, width, channels).
        - segmentation_fn: The segmentation function to use ('slic' by default).
        - segmentation_kwargs: Additional keyword arguments for the segmentation function.
        """
        self.image_shape = image_shape
        self.predictor = self._transform_predictor(predictor)
        self.segmentation_fn = self._get_segmentation_fn(segmentation_fn, segmentation_kwargs)

    def _transform_predictor(self, predictor):
        # Test the predictor output and wrap with ArgmaxTransformer if necessary
        x = np.zeros((1,) + self.image_shape, dtype=np.float32)
        prediction = predictor(x)
        if prediction.ndim == 1 or prediction.shape[1] == 1:
            return predictor  # Predictor returns class labels
        else:
            # Predictor returns probabilities, so wrap it
            return ArgmaxTransformer(predictor)

    def _get_segmentation_fn(self, fn_name, kwargs):
        fn_options = {'slic': slic}
        if fn_name not in fn_options:
            raise ValueError(f"Unsupported segmentation function: {fn_name}")
        if kwargs is None:
            kwargs = {}
        return partial(fn_options[fn_name], **kwargs)

    def explain(self, image, threshold=0.95, p_sample=0.5, tau=0.15, max_anchor_size=None):
        """
        Generate an explanation for the given image.

        Parameters:
        - image: The image to explain.
        - threshold: Precision threshold for anchor acceptance.
        - p_sample: Probability of superpixel being perturbed.
        - tau: Tolerance parameter for the search algorithm.
        - max_anchor_size: Maximum size of the anchor (number of superpixels).

        Returns:
        - An explanation object containing the anchor and related information.
        """
        segments = self._segment_image(image)
        instance_label = self.predictor(image[np.newaxis, ...])[0]

        # Initialize the sampler
        sampler = AnchorImageSampler(
            predictor=self.predictor,
            image=image,
            segments=segments,
            p_sample=p_sample,
            instance_label=instance_label
        )

        # Run the anchor search algorithm
        anchor = self._anchor_search(sampler, threshold, tau, max_anchor_size)

        # Build and return the explanation
        explanation = {
            'image': image,
            'segments': segments,
            'anchor': anchor,
            'precision': threshold
        }
        return explanation

    def _segment_image(self, image):
        # Apply segmentation to the image
        return self.segmentation_fn(image)

    def _anchor_search(self, sampler, threshold, tau, max_anchor_size):
        # Implement the search algorithm to find the best anchor
        # Placeholder implementation (needs to be replaced with actual search logic)
        # For now, we select all superpixels as the anchor
        num_segments = np.unique(sampler.segments).shape[0]
        anchor = list(range(num_segments))
        return anchor


class AnchorImageSampler:
    def __init__(self, predictor, image, segments, p_sample, instance_label):
        """
        Initializes the sampler used for generating perturbed samples.

        Parameters:
        - predictor: The prediction function.
        - image: The original image.
        - segments: The segmentation mask.
        - p_sample: Probability of perturbing a superpixel.
        - instance_label: The predicted label of the original image.
        """
        self.predictor = predictor
        self.image = image
        self.segments = segments
        self.p_sample = p_sample
        self.instance_label = instance_label
        self.num_segments = np.unique(segments).shape[0]

    def sample(self, num_samples):
        """
        Generates perturbed samples.

        Parameters:
        - num_samples: Number of samples to generate.

        Returns:
        - A tuple (perturbed_images, labels), where labels indicate if the perturbed
          image prediction matches the original prediction.
        """
        perturbed_images = []
        labels = []

        for _ in range(num_samples):
            perturbed_image = self._perturb_image()
            perturbed_images.append(perturbed_image)
            pred_label = self.predictor(perturbed_image[np.newaxis, ...])[0]
            labels.append(int(pred_label == self.instance_label))

        return np.array(perturbed_images), np.array(labels)

    def _perturb_image(self):
        """
        Perturbs the image by randomly masking superpixels.

        Returns:
        - A perturbed image.
        """
        mask = np.random.choice([0, 1], size=self.num_segments, p=[self.p_sample, 1 - self.p_sample])
        perturbed_image = copy.deepcopy(self.image)

        for seg_idx in range(self.num_segments):
            if mask[seg_idx] == 0:
                perturbed_image[self.segments == seg_idx] = np.mean(
                    perturbed_image[self.segments == seg_idx], axis=0
                )

        return perturbed_image
