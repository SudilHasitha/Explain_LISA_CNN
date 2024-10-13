from LISA_CNN_ExplainerV3.ExplainLISA import ExplainLISA
from transformers import TFViTForImageClassification, ViTImageProcessor

def test_explain_lisa_with_huggingface_model():
    # Paths to the images
    img_path = 'ai-generated-stray-cat-in-danger-background-animal-background-photo.jpg'
    img_list_paths = ['kitty-cat-kitten-pet-45201.jpeg', 'white-cat-pink-sofa-generate-ai-photo.jpg']

    # Load the model and processor
    model_name = 'google/vit-base-patch16-224'
    model = TFViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)

    # Initialize ExplainLISA
    explainer = ExplainLISA(img_path, model, processor, img_list_paths)

    # Run explanation
    explainer.explain()

if __name__ == "__main__":
    test_explain_lisa_with_huggingface_model()