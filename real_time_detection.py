import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load model
model = load_model('asl_model.h5')
labels = [chr(i) for i in range(65, 75)] + ['NOTHING']  # A–J + NOTHING

# Config
img_size = 64
current_text = ""
final_text = ""
prediction_history = deque(maxlen=10)

# Load wordlist
def load_wordlist(path="wordlist.txt"):
    try:
        with open(path, "r") as f:
            return [line.strip().upper() for line in f if line.strip()]
    except:
        return []

def get_suggestions(prefix, word_list, limit=3):
    if len(prefix) < 1:
        return []
    suggestions = [word for word in word_list if word.startswith(prefix.upper())]
    return suggestions[:limit]

word_list = load_wordlist()

# Start webcam
cap = cv2.VideoCapture(0)
print("📷 Show signs A–J in the green box. Press 'r'=backspace, 'space/enter'=add word, '1-3'=autocomplete, 'q'=quit")

suggestions = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # ROI box
    x1, y1 = 100, 100
    box_size = 300
    x2, y2 = x1 + box_size, y1 + box_size
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (img_size, img_size))
    roi_normalized = roi_resized / 255.0
    roi_input = roi_normalized.reshape(1, img_size, img_size, 3)

    # Prediction
    predictions = model.predict(roi_input)
    confidence = np.max(predictions)
    predicted_label = labels[np.argmax(predictions)] if confidence >= 0.75 else "UNKNOWN"
    prediction_history.append(predicted_label)

    # Add to current word
    if len(prediction_history) == prediction_history.maxlen:
        most_common = max(set(prediction_history), key=prediction_history.count)
        if most_common not in ['UNKNOWN', 'NOTHING']:
            if len(current_text) == 0 or most_common != current_text[-1]:
                current_text += most_common
                prediction_history.clear()

    # UI layout
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, f'{predicted_label} ({confidence*100:.1f}%)', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Text and suggestions box under ROI
    text_box_top = y2 + 20
    text_box_bottom = y2 + 70
    box_width = 400
    box_margin_x = x1
    cv2.rectangle(frame, (box_margin_x, text_box_top), (box_margin_x + box_width, text_box_bottom), (255, 255, 255), -1)

    # Sentence above text box
    cv2.putText(frame, f'Sentence: {final_text.strip()}', (box_margin_x, text_box_top - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Current word
    cv2.putText(frame, f'Text: {current_text}', (box_margin_x + 10, text_box_bottom - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Suggestions
    suggestions = get_suggestions(current_text, word_list)
    for idx, suggestion in enumerate(suggestions):
        cv2.putText(frame, f'{idx + 1}. {suggestion}', (box_margin_x + 10, text_box_bottom + 30 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    # Header bar
    cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 90, 160), -1)
    cv2.putText(frame, '🔠 Real-Time ASL Letter Detection (A–J)', (60, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
    cv2.line(frame, (0, 60), (frame_width, 60), (20, 60, 120), 2)

    cv2.imshow("ASL Word Builder", frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        if len(current_text) > 0:
            current_text = current_text[:-1]
        prediction_history.clear()
    elif key == 32 or key == 13:  # Space or Enter
        if current_text:
            final_text += current_text + " "
            current_text = ""
            prediction_history.clear()
    elif key in [ord('1'), ord('2'), ord('3')]:
        index = int(chr(key)) - 1
        if index < len(suggestions):
            current_text = suggestions[index]

# Cleanup
cap.release()
cv2.destroyAllWindows()
