import cv2
import os

# Define classes: A-Z + NOTHING
classes = [chr(i) for i in range(65,75)] + ['NOTHING']
num_samples = 100
img_size = 64
save_dir = 'mini_train'

# Create folders
os.makedirs(save_dir, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(save_dir, cls), exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
current_class = 0
count = 0

print("📸 Press 'c' to capture an image, 'q' to quit.")

while current_class < len(classes):
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error.")
        break

    frame = cv2.flip(frame, 1)
    cls = classes[current_class]

    # Define green box (ROI)
    x1, y1 = 100, 100
    box_size = 300
    x2, y2 = x1 + box_size, y1 + box_size

    # Draw green rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show class label
    cv2.putText(frame, f'Class: {cls} ({count+1}/{num_samples})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Crop and resize only the ROI
        roi = frame[y1:y2, x1:x2]
        img = cv2.resize(roi, (img_size, img_size))
        path = os.path.join(save_dir, cls, f'{count:03d}.jpg')
        cv2.imwrite(path, img)
        count += 1

        if count >= num_samples:
            print(f"✅ Finished capturing for {cls}")
            current_class += 1
            count = 0

    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
