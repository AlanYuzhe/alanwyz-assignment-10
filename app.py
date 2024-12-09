from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from sklearn.decomposition import PCA

app = Flask(__name__, template_folder="templates")
pca = None

IMAGE_DATABASE = "static/coco_images_resized" 
UPLOAD_FOLDER = "uploads" 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
model.eval()
tokenizer = get_tokenizer('ViT-B-32')
df = pd.read_pickle("image_embeddings.pickle")
file_names = df["file_name"].values
file_names = [os.path.join(IMAGE_DATABASE, fname) for fname in file_names]
image_embeddings = np.stack(df["embedding"].values)


def calculate_image_query(image_path):
    """image"""
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    query_embedding = F.normalize(model.encode_image(image)).detach().cpu().numpy()
    return query_embedding


def calculate_text_query(text_query):
    """Text"""
    text = tokenizer([text_query])
    query_embedding = F.normalize(model.encode_text(text)).detach().cpu().numpy()
    return query_embedding


def calculate_hybrid_query(image_embedding, text_embedding, lam=0.8):
    """Hybrid"""
    image_embedding_tensor = torch.tensor(image_embedding)
    text_embedding_tensor = torch.tensor(text_embedding)
    
    query_embedding = F.normalize(
        lam * text_embedding_tensor + (1 - lam) * image_embedding_tensor
    )
    return query_embedding


def find_similar_images(query_embedding, embeddings_to_use, top_k=5):

    similarities = np.dot(embeddings_to_use, query_embedding.T).flatten()
    print("Similarities shape:", similarities.shape)  # 打印调试信息

    top_indices = np.argsort(-similarities)[:top_k]
    results = [{"image": f"/static/coco_images_resized/{os.path.basename(file_names[i])}",
                "similarity": float(similarities[i])} for i in top_indices]
    return results

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query_type = request.form.get("query-type")  # image, text, or hybrid
    lam = float(request.form.get("lam", 0.8))
    use_pca = request.form.get("use-pca") == "true"
    pca_k = int(request.form.get("pca-k", 50))  
    text_query = request.form.get("text-query")
    image_file = request.files.get("image-query")

    if use_pca:
        global pca
        pca = PCA(n_components=pca_k)
        pca.fit(image_embeddings)
        embeddings_to_use = pca.transform(image_embeddings)
    else:
        embeddings_to_use = image_embeddings

    image_embedding = None
    if image_file:
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
        image_embedding = calculate_image_query(image_path)

    text_embedding = None
    if text_query:
        text_embedding = calculate_text_query(text_query)

    if query_type == "hybrid" and image_embedding is not None and text_embedding is not None:
        query_embedding = calculate_hybrid_query(image_embedding, text_embedding, lam)
    elif query_type == "image" and image_embedding is not None:
        query_embedding = image_embedding
    elif query_type == "text" and text_embedding is not None:
        query_embedding = text_embedding
    else:
        return jsonify({"error": "Invalid query type or missing input."}), 400

    if use_pca:
        query_embedding = pca.transform(query_embedding.reshape(1, -1))

    results = find_similar_images(query_embedding, embeddings_to_use)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=3000)