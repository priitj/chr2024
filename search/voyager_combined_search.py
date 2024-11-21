import os
import numpy as np
from PIL import Image
from voyager import Index, Space

"""
The following resources were used:
- https://spotify.github.io/voyager/python/reference.html
"""

def get_images_to_paths(image_directories, allowed_extensions):
    """Get the paths of all images in the given directory and return the image ids and their paths."""
    images_to_paths = {
        image_path.stem: image_path
        for dir in image_directories
        for image_path in dir.iterdir()
        if image_path.suffix.lower() in allowed_extensions
    }
    return images_to_paths, list(images_to_paths.keys())

def process_image_batch(all_image_ids, start_idx, batch_size, images_to_paths, damaged_image_ids):
    """Process a batch of images, returning their ids and loaded images, while identifying damaged images."""
    batch_image_ids = all_image_ids[start_idx: start_idx + batch_size]
    batch_images = []
    emb_ids = []

    for image_id in batch_image_ids:
        try:
            image = Image.open(images_to_paths[image_id]).convert("RGB")
            image.load()
            batch_images.append(image)
            emb_ids.append(image_id)
        except OSError:
            print(f"\nError processing image {images_to_paths[image_id]}, marking as corrupted.")
            damaged_image_ids.add(image_id)

    return emb_ids, batch_images

def build_voyager_index(ids, vectors, file_path, name, metric=Space.Cosine):
    """Build a Voyager index using the generated vectors."""
    all_vectors = np.array(vectors)
    n_dimensions = all_vectors.shape[1]
    index = Index(space=metric, num_dimensions=n_dimensions, M=50, ef_construction=500)

    index_path = f"{file_path}/{name}.index.voy"
    if os.path.isfile(index_path):
        return index.load(index_path)
    else:
        for i, v in enumerate(all_vectors):
            index.add_item(v, id=int(ids[i]))
        index.save(f"{file_path}/{name}.index.voy")
    return index
