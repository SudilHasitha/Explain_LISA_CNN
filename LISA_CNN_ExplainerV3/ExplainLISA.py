import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shap
from alibi.explainers import AnchorImage, IntegratedGradients
from lime import lime_image
from skimage.segmentation import mark_boundaries
from alibi.utils.visualization import visualize_image_attr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseExplainer:
    def __init__(self, model, image, img_shape=224):
        self.model = model
        self.image = image
        self.img_shape = img_shape

    def preprocess_image(self):
        return np.array(self.image, dtype=np.double)

class AnchorExplainer(BaseExplainer):
    def explain(self, n_segments, compactness, sigma, threshold=.95, p_sample=.5, tau=0.25):
        try:
            predict_fn = lambda x: self.model.predict(x)
            image = np.array(self.image)
            explainer = AnchorImage(predict_fn, image_shape=self.img_shape, segmentation_fn='slic',
                                    segmentation_kwargs={'n_segments': n_segments, 'compactness': compactness, 'sigma': sigma})
            explanation = explainer.explain(image, threshold=threshold, p_sample=p_sample, tau=tau)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
            ax1.imshow(explanation.anchor)
            ax1.axis('off')
            ax2.imshow(explanation.segments)
            ax2.axis('off')
            plt.savefig("AnchorSegmentation.png")
            plt.close(fig)
            
            return explanation.anchor
        except Exception as e:
            logger.error(f"Error in Anchor explanation: {str(e)}")
            return None

class LIMEExplainer(BaseExplainer):
    def explain(self, prediction, positive_only=True, num_features=10000, hide_rest=False, min_weight=0.1):
        try:
            image = np.array(self.image, dtype=np.double)
            explainer = lime_image.LimeImageExplainer(random_state=42)
            explanation = explainer.explain_instance(image, self.model.predict, hide_color=0, num_samples=1000)
            
            image, mask = explanation.get_image_and_mask(
                self.model.predict(image.reshape((1, self.img_shape, self.img_shape, 3))).argmax(axis=1)[0],
                positive_only=positive_only, hide_rest=hide_rest, num_features=num_features, min_weight=min_weight
            )
            
            plt.figure(figsize=(10, 5))
            plt.imshow(mark_boundaries(image, mask))
            plt.axis('off')
            plt.title(f"Prediction: {prediction}")
            plt.savefig("LimeExplanation.png")
            plt.close()
            
            return mask
        except Exception as e:
            logger.error(f"Error in LIME explanation: {str(e)}")
            return None

class IGExplainer(BaseExplainer):
    def explain(self, prediction, n_steps=20, internal_batch_size=20, method="gausslegendre"):
        try:
            ig = IntegratedGradients(self.model, n_steps=n_steps, method=method, internal_batch_size=internal_batch_size)
            instance = np.expand_dims(self.image, axis=0)
            explanation = ig.explain(instance, baselines=None, target=self.model(instance).numpy().argmax(axis=1))
            
            attrs = explanation.attributions[0].squeeze()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            visualize_image_attr(None, self.image, method='original_image', title=f'Prediction {prediction}', plt_fig_axis=(fig, ax1), use_pyplot=False)
            visualize_image_attr(attrs, self.image, method='blended_heat_map', sign='all', show_colorbar=True, title='Overlaid Attributions', plt_fig_axis=(fig, ax2), use_pyplot=True)
            fig.savefig("IGExplanation.png")
            plt.close(fig)
            
            return attrs
        except Exception as e:
            logger.error(f"Error in Integrated Gradients explanation: {str(e)}")
            return None

class SHAPExplainer(BaseExplainer):
    def explain(self, images_list):
        try:
            images_list = np.array(images_list)
            explainer = shap.GradientExplainer(self.model, images_list)
            shap_values = explainer.shap_values(images_list[0:len(images_list)])
            
            shap.image_plot(shap_values, images_list, show=False)
            plt.savefig("SHAPExplanation.png")
            plt.close()
            
            return shap_values
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            return None

class LISAExplainer:
    def __init__(self, model, image, pred_class, img_list, img_shape=224, filter_radius=10):
        self.model = model
        self.image = image
        self.pred_class = pred_class
        self.img_list = img_list
        self.img_shape = img_shape
        self.filter_radius = filter_radius
        self.explanations = {}

    def explain(self):
        self.explanations['Anchor'] = AnchorExplainer(self.model, self.image, self.img_shape).explain(7, 20, 0.5)
        self.explanations['LIME'] = LIMEExplainer(self.model, self.image, self.img_shape).explain(self.pred_class)
        self.explanations['IG'] = IGExplainer(self.model, self.image, self.img_shape).explain(self.pred_class)
        self.explanations['SHAP'] = SHAPExplainer(self.model, self.image, self.img_shape).explain(self.img_list)

    def combine_explanations(self):
        try:
            # Combine explanations (simplified version of the original LISA method)
            combined_mask = np.zeros((self.img_shape, self.img_shape, 3), dtype=np.float32)
            
            if self.explanations['SHAP'] is not None:
                shap_mask = np.where(self.explanations['SHAP'][0][2] > 0, 1, 0)
                combined_mask[:,:,0] += shap_mask
            
            if self.explanations['LIME'] is not None:
                combined_mask[:,:,1] += self.explanations['LIME']
            
            if self.explanations['IG'] is not None:
                combined_mask[:,:,2] += np.mean(np.abs(self.explanations['IG']), axis=-1)
            
            if self.explanations['Anchor'] is not None:
                anchor_mask = cv2.cvtColor(self.explanations['Anchor'], cv2.COLOR_BGR2GRAY)
                combined_mask[:,:,0] += anchor_mask
                combined_mask[:,:,1] += anchor_mask
            
            self.combined_mask = combined_mask / np.max(combined_mask)
            
        except Exception as e:
            logger.error(f"Error in combining explanations: {str(e)}")

    def visualize(self):
        try:
            input_image_array = np.array(self.image * 255., dtype=np.float32)
            
            dst = cv2.addWeighted(input_image_array, 1, self.combined_mask * 255, 0.8, 0)
            dst_I = cv2.bitwise_and(input_image_array, self.combined_mask * 255, input_image_array)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(self.image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            ax2.imshow(dst / 255.)
            ax2.set_title('LISA Explanation (Overlay)')
            ax2.axis('off')
            
            ax3.imshow(dst_I / 255.)
            ax3.set_title('LISA Explanation (Masked)')
            ax3.axis('off')
            
            plt.savefig("LISAExplanation.png")
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")

class ExplainLISA:
    def __init__(self, img, pred_class, img_shape, model, img1, img2, scale=True, filter_radius=10):
        self.img = self.save_load_and_prep(img, int(img_shape), scale)
        self.img1 = self.save_load_and_prep(img1, int(img_shape), scale)
        self.img2 = self.save_load_and_prep(img2, int(img_shape), scale)
        self.model = model
        self.pred_class = pred_class
        self.img_shape = int(img_shape)
        self.img_list = [self.img1, self.img2, self.img]
        self.filter_radius = filter_radius
        self.lisa_explainer = LISAExplainer(self.model, self.img, self.pred_class, self.img_list, self.img_shape, self.filter_radius)

    def explain(self):
        try:
            self.lisa_explainer.explain()
            self.lisa_explainer.combine_explanations()
            self.lisa_explainer.visualize()
        except Exception as e:
            logger.error(f"An error occurred during explanation: {str(e)}")

    @staticmethod
    def save_load_and_prep(img, img_shape, scale):
        try:
            img = tf.io.read_file(img)
            img = tf.io.decode_image(img, channels=3)
            img = tf.image.resize(img, [img_shape, img_shape])
            return img / 255. if scale else img
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return None

# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LISA CNN Explainer')
    parser.add_argument('--img', type=str, required=True, help='Path to the image to explain')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--img_shape', type=int, default=224, help='Image shape')
    parser.add_argument('--filter_radius', type=int, default=10, help='Filter radius')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    explainer = ExplainLISA(args.img, "Unknown", args.img_shape, model, args.img, args.img, filter_radius=args.filter_radius)
    explainer.explain()