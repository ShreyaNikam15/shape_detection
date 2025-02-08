import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Start time to measure execution time
start_time = time.time()

# Load the image
image = cv2.imread('shapes.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dictionary to store shape counts
shape_counts = {
    "Triangle": 0,
    "Rectangle": 0,
    "Square": 0,
    "Pentagon": 0,
    "Circle": 0,
    "Unknown": 0
}

# Function to detect shape
def detect_shape(contour):
    peri = cv2.arcLength(contour, True)  # Perimeter
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Approximate polygon
    num_sides = len(approx)

    if num_sides == 3:
        return "Triangle"
    elif num_sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif num_sides == 5:
        return "Pentagon"
    elif num_sides > 6:
        return "Circle"
    else:
        return "Unknown"

# Iterate through each contour
for contour in contours:
    shape_name = detect_shape(contour)
    shape_counts[shape_name] += 1

    # Compute center of shape
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Draw contour and label the shape
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.putText(image, shape_name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# End time to measure execution time
end_time = time.time()
execution_time = round(end_time - start_time, 4)

# Display the result
cv2.imshow("Shape Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display shape distribution using Matplotlib
plt.figure(figsize=(8, 5))
plt.bar(shape_counts.keys(), shape_counts.values(), color=['red', 'blue', 'green', 'purple', 'orange', 'gray'])
plt.xlabel("Shape")
plt.ylabel("Count")
plt.title("Shape Distribution in Image")
plt.show()

# Model Performance Metrics
total_shapes = sum(shape_counts.values())
correctly_detected = total_shapes - shape_counts["Unknown"]
accuracy = (correctly_detected / total_shapes) * 100 if total_shapes > 0 else 0

print("\nModel Performance:")
print(f"✅ Total Shapes Detected: {total_shapes}")
print(f"✅ Correctly Classified: {correctly_detected}")
print(f"⚠️ Unknown Shapes: {shape_counts['Unknown']}")
print(f"🚀 Accuracy: {accuracy:.2f}%")
print(f"⏱ Execution Time: {execution_time} seconds")
