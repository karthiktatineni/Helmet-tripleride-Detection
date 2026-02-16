import cv2
import numpy as np

try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Test HighGUI", img)
    print("OpenCV HighGUI (imshow) is working.")
    cv2.waitKey(100)
    cv2.destroyAllWindows()
except Exception as e:
    print("OpenCV HighGUI Error:", e)
