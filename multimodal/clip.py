from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
import torch

class CLIPClassifier:
    """Compatible with CLIP and SigLIP
       outputs raw logits only
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.batch_size = 128
        self.setup()
        self.model.to(self.device)

    def setup(self):
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.padding = True

    def empty_cache(self):
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def predictions(self, images, classes, prompts):
        """Predict class probabilities for provided images.
           processes multiple sets of prompts at once
           output is raw logits (must post-process)
        """
    
        # {"image": filename, "true_class": string, "predicted_class": string, "correct": boolean, "predictions": list}
        raw_output = []
    
        for i in range(0, len(images), self.batch_size):
            self.empty_cache()
    
            print("Predicting images " + str(i) + " to " + str(i + self.batch_size) + "...")
            images_batch = images[i:i + self.batch_size]
            inputs = self.processor(text=prompts, images=[image[0] for image in images_batch],
                               return_tensors="pt", padding=self.padding).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            raw_output += outputs.logits_per_image.detach().cpu().numpy().tolist()
    
        image_names = [image[1] for image in images]
        true_classes = [classes.index(image[2]) for image in images]
        return image_names, true_classes, raw_output

    def image_vectors(self, images):
        # images must be a list of images
        all_embeddings = []

        for i in range(0, len(images), self.batch_size):
            self.empty_cache()

            images_batch = images[i:i + self.batch_size]
            inputs = self.processor(images=images_batch,
                               return_tensors="pt", padding=self.padding)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            all_embeddings.extend(outputs.detach().cpu().numpy())

        return all_embeddings

    def text_vectors(self, texts):
        # input must be a list of texts
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            self.empty_cache()

            text_batch = texts[i:i + self.batch_size]
            inputs = self.processor(text=text_batch,
                               return_tensors="pt", padding=self.padding)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            all_embeddings.extend(outputs.detach().cpu().numpy())

        return all_embeddings

class SigLIPClassifier(CLIPClassifier):
    def setup(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.padding = "max_length"
