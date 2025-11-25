import cv2 as cv
import numpy as np
import os
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. CONFIGURATION ---

# 🛑 UPDATE THIS PATH: Directory containing ONLY the new face folders you want to add.
NEW_FACES_DIR = r"E:\face_recognition-20251004T050428Z-1-001\face_recognition\dataset" 
# Name of the existing embeddings file
EMBEDDINGS_FILE = r'E:\face_recognition-20251004T050428Z-1-001\face_recognition\faces_embeddings_done_4classes.npz'
# Name of the existing model file (optional, but good practice to reference)
MODEL_FILE = r'E:\face_recognition-20251004T050428Z-1-001\face_recognition\face_recognition_model.pkl'

TARGET_SIZE = (160, 160)
RANDOM_STATE = 1

# --- 2. INITIALIZATION ---

detector = MTCNN()
embedder = FaceNet()

# --- 3. HELPER FUNCTIONS (From main.ipynb) ---

class FACELOADING:
    """Helper class to extract and load new faces from raw image files."""
    def __init__(self, directory):
        self.directory = directory
        self.target_size = TARGET_SIZE
        self.detector = detector
    
    def extract_face(self, filename):
        img = cv.imread(filename)
        if img is None:
            raise ValueError(f"Could not read image file: {filename}")

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        results = self.detector.detect_faces(img)
        if not results:
            raise ValueError("No face detected in the image.")
            
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr
    
    def load_faces(self):
        FACES = []
        LABELS = []
        print(f"Starting to load NEW faces from: {self.directory}")
        for sub_dir in os.listdir(self.directory):
            dir_path = os.path.join(self.directory, sub_dir)
            if not os.path.isdir(dir_path):
                continue
            
            # Load images for the current sub_dir (class)
            class_faces = []
            for im_name in os.listdir(dir_path):
                path = os.path.join(dir_path, im_name)
                try:
                    single_face = self.extract_face(path)
                    class_faces.append(single_face)
                except Exception as e:
                    # print(f"Skipping file {path}: {e}")
                    pass
            
            if class_faces:
                labels = [sub_dir for _ in range(len(class_faces))]
                print(f"Loaded successfully: {len(labels)} faces for class '{sub_dir}'")
                FACES.extend(class_faces)
                LABELS.extend(labels)
        
        return np.asarray(FACES), np.asarray(LABELS)

def get_embedding(face_img):
    """Generates a 128-D embedding for a 160x160 face image using FaceNet."""
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# --- 4. MAIN UPDATE FUNCTION ---

def incremental_update():
    """Load existing data, add new data, retrain, and save."""
    
    # 4a. Load Existing Data
    print("Loading existing embeddings and labels...")
    try:
        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        old_embeddings = data['arr_0']
        old_labels = data['arr_1']
        print(f"Found {len(old_embeddings)} existing embeddings across {len(np.unique(old_labels))} classes.")
    except FileNotFoundError:
        print(f"Error: Existing embeddings file '{EMBEDDINGS_FILE}' not found.")
        return
    
    # 4b. Load and Embed New Faces
    faceloading = FACELOADING(NEW_FACES_DIR)
    new_faces, new_labels = faceloading.load_faces()

    if new_faces.size == 0:
        print("\nERROR: No new faces were successfully loaded. Stopping update.")
        return

    print("Generating embeddings for new faces...")
    new_embeddings = []
    for img in new_faces:
        new_embeddings.append(get_embedding(img))
    new_embeddings = np.asarray(new_embeddings)
    print(f"Generated {len(new_embeddings)} new embeddings.")
    
    # 4c. Concatenate Datasets
    # Combine old and new embeddings (X)
    X_combined = np.concatenate((old_embeddings, new_embeddings), axis=0)
    # Combine old and new labels (Y)
    Y_combined = np.concatenate((old_labels, new_labels), axis=0)

    print(f"\nTotal combined faces: {len(X_combined)}")
    print(f"Total combined unique classes: {len(np.unique(Y_combined))}")

    # 4d. Re-train LabelEncoder (to include new class names)
    print("Re-training LabelEncoder...")
    encoder = LabelEncoder()
    encoder.fit(Y_combined)
    Y_encoded = encoder.transform(Y_combined)

    # 4e. Split and Train SVC Model
    print("Splitting data and training new SVM model...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_combined, Y_encoded, shuffle=True, random_state=RANDOM_STATE, stratify=Y_encoded # Use stratify to ensure balanced classes
    )

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    print("Model training complete.")

    # 4f. Evaluate and Save
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_preds)
    print(f"\nUpdated Model Accuracy: {accuracy:.4f}")

    print("Saving updated model, encoder, and combined embeddings file...")
    
    # Save combined embeddings and labels (OVERWRITES old NPZ)
    np.savez_compressed(EMBEDDINGS_FILE, X_combined, Y_combined)
    print(f"✅ Combined embeddings and labels saved to {EMBEDDINGS_FILE}")
    
    # Save updated encoder and model (OVERWRITES old PKL files)
    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
    print("✅ Updated encoder saved to encoder.pkl")
    
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(model, file)
    print(f"✅ Updated model saved to {MODEL_FILE}")

if __name__ == "__main__":
    incremental_update()