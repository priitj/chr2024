import torch
import numpy as np
from lavis.models import load_model_and_preprocess


class Blip2Classifier:
    """Blip-2 classification model
       outputs raw logits only
    """

    def __init__(self, model_name="blip2", model_type="pretrain_vitL", device="cuda"):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                                         model_type=model_type,
                                                                                         is_eval=True,
                                                                                         device=device)
        self.batch_size = 128

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

        image_batch = [image[0] for image in images]
        image_vectors = self.image_vectors(image_batch).detach().cpu().numpy()
        text_vectors = self.text_vectors(prompts).detach().cpu().numpy()

        text_T = text_vectors.T / np.linalg.norm(text_vectors, axis=1)
        image_arr = image_vectors / np.linalg.norm(image_vectors, axis=1).reshape(1, -1).T

        raw_output = image_arr.dot(text_T).tolist()

        image_names = [image[1] for image in images]
        true_classes = [classes.index(image[2]) for image in images]
        return image_names, true_classes, raw_output

    def image_vectors(self, images):
        # images must be a list of images
        images_embedding = None

        for image in images:
            image_processed = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            sample = {"image": image_processed, "text_input": None}

            image_emb = self.model.extract_features(sample, mode="image").image_embeds[:, 0, :]  # size (1, 768)

            # stack tensor
            if images_embedding is None:
                images_embedding = image_emb
            else:
                images_embedding = torch.cat((images_embedding, image_emb), 0)

        return images_embedding.detach().cpu()

    def text_vectors(self, texts):
        # input must be a list of texts
        texts_embedding = None

        for text in texts:
            text_input = self.txt_processors["eval"](text)
            sample = {"image": None, "text_input": [text_input]}

            text_emb = self.model.extract_features(sample, mode="text").text_embeds[:, 0, :]  # size (1, 768)

            # stack tensor
            if texts_embedding is None:
                texts_embedding = text_emb
            else:
                texts_embedding = torch.cat((texts_embedding, text_emb), 0)

        return texts_embedding.detach().cpu()
