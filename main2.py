import cv2
import numpy as np
import sys
import time
import math
from skimage.feature import canny
from skimage.measure import EllipseModel, ransac
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)

    # Display results
    cv2.imshow("Results", im)



color = cv2.imread("/Users/florianscalvini/Downloads/logo-ellipse.png")
img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

edges2 = canny(img, sigma=3)
edges_images = (edges2 * 255).astype(np.uint8)
point = np.where
model = EllipseModel()
point = np.asarray(np.where(edges2 == True))
model.estimate(point.transpose())
x0, y0, a, b, phi = model.params
cv2.ellipse(color, (int(round(y0)),int(round(x0))), (int(round(b)),int(round(a))), int(round(math.degrees(phi))), 0, 360, (255,0,255), thickness=5, lineType=cv2.LINE_AA)
while(True):
    cv2.imshow('RealSense', color)
    cv2.waitKey(1)



