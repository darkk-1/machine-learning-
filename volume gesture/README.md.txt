# ✋ Gesture-Based Volume Control using OpenCV and MediaPipe

## 🎯 Overview
This project allows users to control their **system volume** using **hand gestures** detected by a webcam.  
It uses **OpenCV** for real-time video capture, **MediaPipe** for hand tracking, and **PyCaw** for Windows audio control.

---

## ⚙️ How It Works
1. The webcam captures live video.
2. MediaPipe detects hand landmarks in each frame.
3. The distance between the **thumb tip** and **pinky tip** is calculated.
4. This distance is mapped to the system’s audio range.
5. Moving your fingers closer or farther apart decreases or increases the volume respectively.

---

## 🧠 Dependencies
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- PyCaw
- comtypes

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/gesture_volume_control.git
cd gesture_volume_control
pip install -r requirements.txt
▶️ Run the Project
bash
Copy code
python volume.py
Press 'q' to quit the application.

🧩 Requirements File Example
ini
Copy code
opencv-python==4.10.0.84
mediapipe==0.10.9
numpy==2.1.1
pycaw==20240210
comtypes==1.4.6
📸 Example Output
The webcam window shows:

Hand landmarks (connected points)

Volume bar indicating current level

Volume percentage displayed in real-time

🧑‍💻 Author
Dinesh Kumar M
📧 dk895361@gmail.com
🔗 LinkedIn

📜 License
Open-source under the MIT License.

yaml
Copy code

---

✅ Once saved, this file will automatically render beautifully on GitHub with headers, code blocks, and emojis.  
Would you like me to give you a `.gitignore` file next (for this project)?