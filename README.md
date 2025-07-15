# asl-sign-language-detector

This project uses computer vision and deep learning to recognize American Sign Language (ASL) hand signs for the letters **A to J** in real time. It builds words and sentences from hand gestures captured by a webcam, with smart autocomplete and UI overlays — all using Python.



---

## 📌 Features

- ✅ Real-time detection of ASL letters
- 🟩 Live webcam with green detection box
- 🧠 Smart text prediction using a trained model
- 💡 Word suggestions based on typed letters
- ⌨️ Keyboard shortcuts to autocomplete or edit
- 📝 Sentence builder with finalized word history

---

## 🛠 Tech Stack

| Tool        | Purpose                     |
|-------------|-----------------------------|
| Python      | Core programming language   |
| OpenCV      | Webcam capture & GUI        |
| TensorFlow / Keras | Deep learning model     |
| NumPy       | Array manipulation          |
| Pillow      | Image handling (optional)   |

ASL_Project/
│
├── capture_dataset.py # For collecting hand sign images
├── train_model.py # For training CNN on A–J signs
├── real_time_detection.py # Final app with UI + prediction
├── asl_model.h5 # Trained Keras model (not uploaded)
├── wordlist.txt # Word dictionary for suggestions
├── requirements.txt # Python dependencies
├── README.md # Project overview
├── mini_train/ # Dataset folder (optional)

Install dependencies:
pip install -r requirements.txt

Run the app:
python real_time_detection.py

Key	Action:
1–3 	Select one of the suggested words
r	    Delete last letter (Backspace)
space or enter	Finalize word, add to sentence
q	    Quit the app
