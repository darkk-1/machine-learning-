
# 🌸 Iris Flower Classification using Decision Tree

## 📘 Overview
This project implements a **Decision Tree Classifier** to predict the species of Iris flowers — *Setosa*, *Versicolor*, or *Virginica* — based on four key features:  
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

The model is trained and evaluated using the **Iris dataset**, one of the most well-known datasets in machine learning.  

---

## 🧠 Project Objectives
- Understand and visualize the Iris dataset.  
- Preprocess data and split it into training and testing sets.  
- Train a **Decision Tree Classifier** for flower species prediction.  
- Evaluate the model using accuracy, confusion matrix, and classification report.  
- Visualize the trained decision tree for interpretability.

---

## 📂 Project Structure

iris_decision_tree_project/  
│
├── notebooks/
│   └── Iris_Flower_Classification_using_Supervised_Machine_Learning_.ipynb            
│
├── model/
│   └── decision_tree_model.pkl       # Saved trained model (optional)
│
├── requirements.txt                  # List of dependencies
├── README.md                         # Project documentation
├── model/
│   └── architecture_diagram.png          # Model architecture visualization

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/iris_decision_tree_project.git
cd iris_decision_tree_project

2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt


🚀 How to Run the Project
Option 1: Run Jupyter Notebook
jupyter notebook notebooks/Iris_Flower_Classification_using_Supervised_Machine_Learning_.ipynb

Option 2: Run as Python scripts
python src/train_model.py
python src/evaluate_model.py


📊 Model Architecture
The Decision Tree Classifier splits the dataset based on feature thresholds to minimize impurity (using Gini Index or Entropy).
Simplified Workflow:
graph TD
    A[Input Features: Sepal & Petal measurements] --> B[Data Preprocessing]
    B --> C[Decision Tree Training]
    C --> D[Model Evaluation (Accuracy, Confusion Matrix)]
    D --> E[Prediction: Iris Setosa / Versicolor / Virginica]


📈 Results


Accuracy: ~95–98% (varies by train-test split)


Confusion Matrix: Displays true vs. predicted classes.


Classification Report: Precision, recall, and F1-score for each class.



🧩 Dependencies


Python 3.8+


pandas


numpy


scikit-learn


matplotlib


seaborn


joblib


(see requirements.txt for exact versions)

📜 License
This project is open-source under the MIT License. Feel free to use, modify, and share.

👨‍💻 Author
Dinesh Kumar M
📧 dk895361@gmail.com
🔗 LinkedIn

---

Would you like me to generate this as a downloadable **`README.md` file** (like before)?

