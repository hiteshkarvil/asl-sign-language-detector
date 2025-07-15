import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from collections import deque

# Load model and labels
model = load_model('asl_model.h5')
labels = [chr(i) for i in range(65, 75)] + ['NOTHING']  # A–J + NOTHING

# Globals
img_size = 64
current_text = ""
prediction_history = deque(maxlen=10)

# Tkinter setup
root = tk.Tk()
root.title("🤟 ASL Real-Time Recognition")
root.geometry("1000x600")
root.configure(bg="#f0f0f0")

# Frames
left_frame = tk.Frame(root, width=640, height=480, bg="black")
left_frame.pack(side="left", padx=10, pady=10)
right_frame = tk.Frame(root, width=300, height=480, bg="white")
right_frame.pack(side="right", padx=10, pady=10)

# Webcam label
video_label = tk.Label(left_frame)
video_label.pack()

# Word display label
word_label = tk.Label(right_frame, text="Text: ", font=("Helvetica", 20), bg="white", anchor="w", justify="left")
word_label.pack(pady=30, padx=10, anchor="nw")

# Reset button
def reset_text():
    global current_text
    current_text = ""
    word_label.config(text="Text: ")

reset_btn = tk.Button(right_frame, text="🔄 Reset", font=("Helvetica", 14), command=reset_text)
reset_btn.pack(pady=10)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

def update_frame():
    global current_text

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    x1, y1, box_size = 100, 100, 300
    x2, y2 = x1 + box_size, y1 + box_size

    # ROI & prediction
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (img_size, img_size))
    roi_normalized = roi_resized / 255.0
    roi_input = roi_normalized.reshape(1, img_size, img_size, 3)

    predictions = model.predict(roi_input)
    confidence = np.max(predictions)
    predicted_label = labels[np.argmax(predictions)] if confidence >= 0.75 else "UNKNOWN"

    prediction_history.append(predicted_label)

    # Add to text if stable
    if len(prediction_history) == prediction_history.maxlen:
        most_common = max(set(prediction_history), key=prediction_history.count)
        if most_common not in ['UNKNOWN', 'NOTHING']:
            if len(current_text) == 0 or most_common != current_text[-1]:
                current_text += most_common
                word_label.config(text=f"Text: {current_text}")
                prediction_history.clear()

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{predicted_label} ({confidence*100:.1f}%)', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Convert image for tkinter
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Start loop
update_frame()
root.mainloop()

# Release cam on close
cap.release()
cv2.destroyAllWindows()
