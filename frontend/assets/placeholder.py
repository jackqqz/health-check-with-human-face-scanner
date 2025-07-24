# Save this in a separate Python script to create the placeholder image
import cv2
import numpy as np
import os

# Create a simple placeholder image
placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 255
# Draw a simple face-like shape
cv2.circle(placeholder, (50, 50), 30, (200, 200, 200), -1)  # Face
cv2.circle(placeholder, (40, 40), 5, (0, 0, 0), -1)  # Left eye
cv2.circle(placeholder, (60, 40), 5, (0, 0, 0), -1)  # Right eye
cv2.line(placeholder, (40, 60), (60, 60), (0, 0, 0), 2)  # Mouth

# Save the placeholder
save_path = os.path.join("placeholder.jpg")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, placeholder)
print(f"Placeholder saved to {save_path}")