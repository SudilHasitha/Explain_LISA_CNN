import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from LISA_CNN_ExplainerV3.utils.shap_explanation import SHAPGradientExplainer
from LISA_CNN_ExplainerV3.utils.integrated_gradient import IntegratedGradients
from LISA_CNN_ExplainerV3.utils.anchor_explanation import AnchorImageExplainer
from LISA_CNN_ExplainerV3.utils.lime_explanation import LIMEImageExplainer
from skimage.segmentation import mark_boundaries
import logging
import argparse
from transformers import TFPreTrainedModel, AutoConfig
import tf_keras
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseExplainer:
    def __init__(self, model, image, processor):
        self.model = model
        self.image = image
        self.processor = processor

    def preprocess_image(self):
        return self.processor(images=self.image, return_tensors='tf')['pixel_values'][0]

class AnchorExplainer(BaseExplainer):
    def explain(self, n_segments, compactness, sigma, threshold=.95, p_sample=.5, tau=0.25):
        try:
            predict_fn = lambda x: self.model.predict(self.processor(images=x, return_tensors='tf')['pixel_values'])
            image = np.array(self.image)
            explainer = AnchorImageExplainer(predict_fn, image_shape=image.shape, segmentation_fn='slic',
                                             segmentation_kwargs={'n_segments': n_segments, 'compactness': compactness, 'sigma': sigma})
            explanation = explainer.explain(image, threshold=threshold, p_sample=p_sample, tau=tau)
            
            # Convert to NumPy arrays
            anchor_mask = np.array(explanation['anchor'])
            segments = np.array(explanation['segments'])

            # Ensure shapes are compatible
            if anchor_mask.shape != image.shape[:2]:
                anchor_mask = anchor_mask.reshape(image.shape[:2])

            # Visualize the explanation
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(mark_boundaries(image / 255., anchor_mask))
            ax.axis('off')
            ax.set_title('Anchor Explanation')
            plt.savefig("AnchorSegmentation.png")
            plt.close(fig)
            
            return anchor_mask
        except Exception as e:
            logger.error(f"Error in Anchor explanation: {str(e)}")
            return None

class LIMEExplainer(BaseExplainer):
    def explain(self, positive_only=True, num_features=1000, hide_rest=False, min_weight=0.1):
        try:
            image = np.array(self.image)
            explainer = LIMEImageExplainer(random_state=42)
            predict_fn = lambda x: self.model.predict(self.processor(images=x, return_tensors='tf')['pixel_values'])
            explanation = explainer.explain_instance(image, predict_fn, hide_color=0, num_samples=500)
            
            # Get the prediction label index
            prediction = np.argmax(predict_fn([image]), axis=1)[0]
            
            image_vis, mask = explanation.get_image_and_mask(
                label=prediction,
                positive_only=positive_only,
                hide_rest=hide_rest,
                num_features=num_features,
                min_weight=min_weight
            )
            
            plt.figure(figsize=(10, 5))
            plt.imshow(mark_boundaries(image_vis / 255., mask))
            plt.axis('off')
            plt.title(f"LIME Explanation (Prediction: {prediction})")
            plt.savefig("LimeExplanation.png")
            plt.close()
            
            return mask
        except Exception as e:
            logger.error(f"Error in LIME explanation: {str(e)}")
            return None


class IGExplainer(BaseExplainer):
    def explain(self, n_steps=50, internal_batch_size=32, method="gausslegendre"):
        try:
            # Preprocess the image to get pixel values
            input_dict = self.processor(images=self.image, return_tensors='tf')
            image = input_dict['pixel_values']  # Shape: (1, height, width, channels)
            baseline = tf.zeros_like(image)
            
            ig = IntegratedGradients(self.model, n_steps=n_steps, method=method, internal_batch_size=internal_batch_size)
            attributions = ig.explain(inputs=image, baselines=baseline)
            attributions = attributions[0].numpy()

            # Visualization
            attribution_sum = np.sum(np.abs(attributions), axis=-1).squeeze()
            plt.figure(figsize=(10, 5))
            plt.imshow(attribution_sum, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title("Integrated Gradients Explanation")
            plt.savefig("IGExplanation.png")
            plt.close()
            
            return attributions
        except Exception as e:
            logger.error(f"Error in Integrated Gradients explanation: {str(e)}")
            return None

class SHAPExplainer(BaseExplainer):
    def explain(self, images_list):
        try:
            # Preprocess images and convert to tensors
            images_tensor_list = [self.processor(images=img, return_tensors='tf')['pixel_values'][0] for img in images_list]
            images_tensor = tf.stack(images_tensor_list)  # Shape: (batch_size, height, width, channels)

            # Use up to 10 background samples
            background = images_tensor[:10]

            # Create SHAP Gradient Explainer
            explainer = SHAPGradientExplainer(self.model.model, background)  # Pass the original model

            # Explain the first image
            shap_values = explainer.explain(images_tensor[0:1])

            # Visualization
            shap_value = shap_values[0].numpy()
            original_image = images_tensor[0].numpy()
            explainer.visualize(shap_value, original_image)
            plt.savefig("SHAPExplanation.png")
            plt.close()
            
            return shap_values
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            return None

class LISAExplainer:
    def __init__(self, model, image, processor, pred_class, img_list):
        self.model = model
        self.image = image
        self.processor = processor
        self.pred_class = pred_class
        self.img_list = img_list
        self.explanations = {}

    def explain(self):
        # Generate explanations using the respective explainer classes
        self.explanations['Anchor'] = AnchorExplainer(self.model, self.image, self.processor).explain(7, 20, 0.5)
        self.explanations['LIME'] = LIMEExplainer(self.model, self.image, self.processor).explain(self.pred_class)
        self.explanations['IG'] = IGExplainer(self.model, self.image, self.processor).explain(self.pred_class)
        self.explanations['SHAP'] = SHAPExplainer(self.model, self.image, self.processor).explain(self.img_list)

    def combine_explanations(self, method_weights=None):
        """
        Combine explanations from different methods mathematically.

        Parameters:
        - method_weights: Optional dictionary specifying weights for each method.
                          Keys should be 'Anchor', 'LIME', 'IG', 'SHAP'.
                          Values should be non-negative and sum to 1.
                          If None, equal weights are assigned.
        """
        try:
            # List of methods
            methods = ['Anchor', 'LIME', 'IG', 'SHAP']
            K = len(methods)

            # Assign equal weights if not provided
            if method_weights is None:
                method_weights = {method: 1 / K for method in methods}

            # Normalize weights
            total_weight = sum(method_weights.values())
            method_weights = {method: w / total_weight for method, w in method_weights.items()}

            # Initialize combined attribution map
            combined_attribution = np.zeros((self.image.size[1], self.image.size[0]), dtype=np.float32)

            # Process each method
            for method in methods:
                attribution = None

                if self.explanations.get(method) is not None:
                    if method == 'SHAP':
                        # SHAP values
                        shap_values = self.explanations['SHAP'][0]  # Assuming first element is the SHAP values
                        # Aggregate over channels if necessary
                        if shap_values.ndim == 3:
                            shap_attribution = np.abs(shap_values).mean(axis=-1)
                        else:
                            shap_attribution = np.abs(shap_values)
                        # Normalize
                        shap_norm = shap_attribution / np.max(shap_attribution) if np.max(shap_attribution) != 0 else shap_attribution
                        attribution = shap_norm

                    elif method == 'LIME':
                        # LIME mask
                        lime_mask = self.explanations['LIME']
                        lime_attribution = lime_mask.astype(np.float32)
                        # Normalize
                        lime_norm = lime_attribution / np.max(lime_attribution) if np.max(lime_attribution) != 0 else lime_attribution
                        attribution = lime_norm

                    elif method == 'IG':
                        # Integrated Gradients attributions
                        ig_attribution = np.abs(self.explanations['IG']).mean(axis=-1)
                        # Normalize
                        ig_norm = ig_attribution / np.max(ig_attribution) if np.max(ig_attribution) != 0 else ig_attribution
                        attribution = ig_norm

                    elif method == 'Anchor':
                        # Anchor mask
                        anchor_mask = self.explanations['Anchor']
                        # Convert to grayscale if necessary
                        if anchor_mask.ndim == 3 and anchor_mask.shape[2] == 3:
                            anchor_attribution = cv2.cvtColor(anchor_mask, cv2.COLOR_BGR2GRAY)
                        else:
                            anchor_attribution = anchor_mask
                        # Normalize
                        anchor_norm = anchor_attribution / np.max(anchor_attribution) if np.max(anchor_attribution) != 0 else anchor_attribution
                        attribution = anchor_norm

                    # Update combined attribution
                    if attribution is not None:
                        combined_attribution += method_weights[method] * attribution

            # Normalize the combined attribution to [0, 1]
            self.combined_mask = combined_attribution / np.max(combined_attribution) if np.max(combined_attribution) != 0 else combined_attribution

        except Exception as e:
            logger.error(f"Error in combining explanations: {str(e)}")

    def visualize(self):
        try:
            input_image_array = np.array(self.image, dtype=np.float32)
            if input_image_array.max() > 1.0:
                input_image_array /= 255.0  # Scale to [0, 1] if necessary

            # Ensure combined_mask has the correct shape
            if self.combined_mask.shape != input_image_array.shape[:2]:
                combined_mask_resized = cv2.resize(self.combined_mask, (input_image_array.shape[1], input_image_array.shape[0]))
            else:
                combined_mask_resized = self.combined_mask

            # Create a color map for visualization
            combined_mask_rgb = np.stack([combined_mask_resized]*3, axis=-1)  # Convert to 3 channels if necessary

            # Overlay the combined mask onto the original image
            overlay = cv2.addWeighted(input_image_array, 1, combined_mask_rgb, 0.5, 0)
            masked_image = input_image_array * combined_mask_rgb

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(input_image_array)
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2.imshow(overlay)
            ax2.set_title('LISA Explanation (Overlay)')
            ax2.axis('off')

            ax3.imshow(masked_image)
            ax3.set_title('LISA Explanation (Masked)')
            ax3.axis('off')

            plt.savefig("LISAExplanation.png")
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, inputs):
        outputs = self.model(inputs)
        # Convert logits to probabilities
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        return probs.numpy()

    def __call__(self, inputs):
        # Return the logits directly
        return self.model(inputs).logits


class ExplainLISA:
    def __init__(self, img_path, model, processor, img_list_paths):
        self.img = Image.open(img_path).convert('RGB')
        self.img_list = [Image.open(p).convert('RGB') for p in img_list_paths]
        self.model = ModelWrapper(model)
        self.processor = processor

        # Get the predicted class
        input_tensor = self.processor(images=self.img, return_tensors='tf')['pixel_values']
        predictions = self.model.predict(input_tensor)
        self.pred_class = np.argmax(predictions, axis=1)[0]

        self.lisa_explainer = LISAExplainer(self.model, self.img, self.processor, self.pred_class, self.img_list)

    def explain(self):
        try:
            self.lisa_explainer.explain()
            self.lisa_explainer.combine_explanations()
            self.lisa_explainer.visualize()
        except Exception as e:
            logger.error(f"An error occurred during explanation: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='LISA CNN Explainer')
    parser.add_argument('--img', type=str, required=True, help='Path to the image to explain')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model or Hugging Face model identifier')
    args = parser.parse_args()

    try:
        from transformers import TFViTForImageClassification, ViTImageProcessor
        model = TFViTForImageClassification.from_pretrained(args.model)
        processor = ViTImageProcessor.from_pretrained(args.model)
        logger.info(f"Loaded Hugging Face model '{args.model}' in TensorFlow.")
    except Exception as e:
        logger.error(f"Failed to load model '{args.model}' from Hugging Face: {str(e)}")
        return

    # Prepare the image list (using the same image for simplicity)
    img_list_paths = [args.img, args.img]

    # Initialize and run ExplainLISA
    explainer = ExplainLISA(args.img, "Unknown", model, processor, img_list_paths)
    explainer.explain()

if __name__ == "__main__":
    main()