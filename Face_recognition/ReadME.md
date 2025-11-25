# Face Recognition System (FaceNet + SVM)

This repository contains the code and necessary files for a face recognition system implemented using the **FaceNet** model for generating high-dimensional face embeddings and a **Support Vector Machine (SVM)** classifier for final classification. The system is designed for both initial training via a Jupyter notebook and real-time inference via Python scripts.

## 🚀 Key Features

* **Face Extraction:** Uses the **MTCNN** (Multi-task Cascaded Convolutional Networks) for accurate face and bounding box detection during training and the **Haar Cascade** detector for fast detection during real-time inference.
* **Feature Encoding:** Utilizes the pre-trained **FaceNet** model to generate a 128-dimensional embedding for each face.
* **Classification:** A **Linear Support Vector Machine (SVC)** is trained on the embeddings to classify faces into known identities.
* **Incremental Update:** Includes a dedicated script (`update.py`) to easily add new faces and retrain the model without re-processing the entire dataset.

---

## 📁 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| **`main.ipynb`** | The primary Jupyter notebook detailing the entire workflow: data loading, face extraction, embedding generation, SVM training, and final testing. |
| **`realtime.py`** | A Python script for live face recognition using a video file or webcam (configured for video in the provided code snippet). |
| **`update.py`** | A utility script to **add new faces** (images) to the existing database, combine embeddings, and retrain the model and encoder. |
| **`requirements.txt`** | Lists all required Python packages for environment setup. |
| **`faces_embeddings_done_4classes.npz`** | Saved NumPy file containing the **FaceNet embeddings** and corresponding **labels** used for training the classifier. |
| **`svm_model_160x160.pkl`** | The final **trained SVC model** (the classifier). This file is loaded by `realtime.py`. |
| **`haarcascade_frontalface_default.xml`** | OpenCV's pre-trained classifier used for fast face detection in the `realtime.py` script. |
| **`face_recognition_model.pkl`** | An alternative or previous version of the saved trained SVC model. |
| **`Photo-Based Face Verification in Video Using FaceNet.py`** | A script for verifying if a single target person appears in a video based on cosine similarity. |
| **`encoder.pkl`** *(Crucial)* | A saved `LabelEncoder` object, which maps numerical class IDs back to actual names/labels. Must be included for inference. |

---

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd face-recognition-project
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. Install all necessary packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Preparation (Crucial):**
    The training process requires a directory of raw images structured as follows:
    ```
    /Original Images/
        /person_A_name/
            img_01.jpg
            img_02.jpg
        /person_B_name/
            img_01.jpg
    ```
    Ensure the path in `main.ipynb` and `update.py` points to this root directory.

---

## 🚀 How to Use

### 1. Initial Training and Model Generation

Use the `main.ipynb` notebook to run the full training pipeline, which generates and saves the necessary `.npz` and `.pkl` files.

### 2. Real-Time Recognition (Video/Webcam)

Run the `realtime.py` script:

```bash
python realtime.py

### 3. Updating the Model with New Faces

To add new individuals to your recognition database:

Place the new face images into class-named folders in a new, separate directory (e.g., /NewFacesToEmbed/).

Update the NEW_FACES_DIR path inside the update.py script to point to this new directory.

Run the update script: