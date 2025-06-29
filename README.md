# Fashion-Recommender-
This project uses a pre-trained **ResNet50** model to build an image-based recommendation engine. It extracts deep features from images and recommends visually similar images using **Euclidean distance** and **Nearest Neighbors**.

---




##  How It Works

1. **Feature Extraction Phase**:
   - Load a pre-trained ResNet50 model with `imagenet` weights.
   - Pass uploaded images through the model to extract feature vectors.
   - Normalize and save the vectors to `embeddings.pkl` and corresponding filenames to `filenames.pkl`.

2. **Recommendation Phase**:
   - Load stored embeddings and filenames.
   - Upload a query image.
   - Compute its feature vector using ResNet50.
   - Use `NearestNeighbors` from scikit-learn to find and display the 5 most similar images.





---

## Environment Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- OpenCV (`cv2`)
- Scikit-learn
- Google Colab (for file upload interface)

---

## Installation & Usage (on Google Colab)

1. **Upload your image dataset:**
   ```python
   uploaded_files = files.upload()
