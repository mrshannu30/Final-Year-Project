# Final-Year-Project
This repository contains my Final Year Project, focused on developing a practical real-world application. It uses backend logic, database management, and a user-friendly interface to solve a specific problem. The project helped me apply theory, build technical skills, and gain hands-on development experience.

## Face recognition with ML (Python)
This project includes a minimal machine-learning-based face recognition pipeline using Python, `face_recognition`, and a linear SVM classifier. It expects a training directory with one subfolder per person and supports prediction with optional annotated outputs.

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare training data
```
train/
  Alice/
    alice_1.jpg
    alice_2.jpg
  Bob/
    bob_1.jpg
```

### 3) Train the model
```bash
python face_recognition_ml.py --train-dir train --model-out face_model.pkl
```

### 4) Run prediction and save annotation
```bash
python face_recognition_ml.py \
  --predict sample.jpg \
  --model-out face_model.pkl \
  --annotate-out annotated.jpg \
  --threshold 0.6
```
