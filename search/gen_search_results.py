import json
import os.path
from pathlib import Path
import numpy as np

from search.voyager_combined_search import build_voyager_index, get_images_to_paths, process_image_batch
from utils.device_config import configure_device

K = 30
INDEX = True

def process_images(image_directory, subfolders, vectorizer, model_name):
    """Build images index."""
    image_directory = Path(image_directory)
    image_paths = [Path(f"{image_directory}/{folder}") for folder in subfolders]

    allowed_extensions = {".jpeg", ".jpg", ".png", ".webp"}
    image_to_path, image_ids = get_images_to_paths(image_paths, allowed_extensions)

    emb_ids, embeddings = generate_embeddings(image_ids, image_to_path, vectorizer)
    if INDEX:
        print("Building Voyager index...")
        voyager_embeddings_index = build_voyager_index(emb_ids, embeddings, image_directory, model_name)
    
        return emb_ids, voyager_embeddings_index
    else:
        return emb_ids, embeddings

def generate_embeddings(all_image_ids, images_to_paths, vectorizer):
    """Generate CLIP embeddings for all images"""
    emb_ids = []
    embeddings = []
    damaged_ids = set()
    image_batch_size = 128 # how many images to load into memory
                           # does not affect vectorizer batch size
    for i in range(0, len(all_image_ids), image_batch_size):
        batch_image_ids, batch_images = process_image_batch(all_image_ids,
                        i, image_batch_size, images_to_paths, damaged_ids)

        emb_ids += batch_image_ids
        embeddings += vectorizer.image_vectors(batch_images)
    print("Damaged image ids", damaged_ids)
    return emb_ids, embeddings

def search_by_texts(texts, vectorizer, index, k):
    """Find top-k results for multiple search strings"""

    text_vectors = vectorizer.text_vectors(texts)
    outputs = {}
    for i, text in enumerate(texts):
        text_vector = text_vectors[i]
        neighbor_ids, distances = index.query(text_vector, k)
        outputs[text] = [(int(n_id), float(d))
                         for n_id, d in zip(neighbor_ids, distances)]

    return outputs

def search_by_dot(texts, vectorizer, emb_ids, embeddings, k):
    """Find top-k results by dot product of embeddings"""

    text_vectors = vectorizer.text_vectors(texts)
    image_arr = np.array(embeddings)
    image_T = image_arr.T / np.linalg.norm(image_arr, axis=1)
    outputs = {}
    for i, text in enumerate(texts):
        text_vector = np.array(text_vectors[i]).reshape(1, -1)
        text_vector /= np.linalg.norm(text_vector)
        sim = text_vector.dot(image_T)
        topn = np.argsort(-sim, axis=1)
        outputs[text] = [(int(emb_ids[j]), float(sim[0, j])) for j in topn[0, :k]]

    return outputs

def run_searches(model_name, image_directory, subfolders, searches_file, out_dir):
    if "openai/clip" in model_name:
        from multimodal.clip import CLIPClassifier
        vectorizer = CLIPClassifier(model_name, device=device)
    elif "google/siglip" in model_name:
        from multimodal.clip import SigLIPClassifier
        vectorizer = SigLIPClassifier(model_name, device=device)
    elif "blip2" in model_name:
        from multimodal.blip2 import Blip2Classifier
        vectorizer = Blip2Classifier(model_name, device=device)
    else:
        raise NotImplementedError("model name not known")
    sanitized_model_name = model_name.replace("/", "_")
    if not INDEX:
        sanitized_model_name += "_noindex"

    if INDEX:
        _, index = process_images(image_directory, subfolders, vectorizer, sanitized_model_name)
    else:
        emb_ids, embeddings = process_images(image_directory, subfolders, vectorizer, sanitized_model_name)
    search_strings = set()
    with open(searches_file) as f:
        search_sets = json.load(f)
    for k, v in search_sets.items():
        for w in v:
            search_strings.add(w)

    if INDEX:
        outputs = search_by_texts(list(search_strings), vectorizer, index, k=K)
    else:
        outputs = search_by_dot(list(search_strings), vectorizer, emb_ids, embeddings, k=K)

    results = {"model_name": model_name,
               "index": "Voyager" if INDEX else "none",
               "results": outputs}
    out_file = os.path.join(out_dir,
                           "{}_search.json".format(sanitized_model_name))
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import sys
    device = configure_device()
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    model_name = sys.argv[1]
    image_directory = sys.argv[2]
    searches_file = sys.argv[3]
    out_dir = sys.argv[4]
    # TODO: make configurable
    # currently hardcoded because this is the only experiment needed
    subfolders = ["photos"]
    #subfolders = ["photos3"]

    run_searches(model_name, image_directory, subfolders, searches_file, out_dir)
