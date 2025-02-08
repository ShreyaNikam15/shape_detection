import cv2
import numpy as np

# Create a blank white image
image = np.ones((500, 500, 3), dtype="uint8") * 255

# Draw a rectangle
cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)

# Draw a circle
cv2.circle(image, (300, 100), 50, (0, 0, 0), -1)

# Draw a triangle
pts = np.array([[250, 200], [200, 300], [300, 300]], np.int32)
cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 0), thickness=5)

# Draw a pentagon
pentagon = np.array([[400, 200], [350, 250], [370, 320], [430, 320], [450, 250]], np.int32)
cv2.polylines(image, [pentagon], isClosed=True, color=(0, 0, 0), thickness=5)

# Save the image
cv2.imwrite("shapes.png", image)

# Display the image
cv2.imshow("Generated Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
