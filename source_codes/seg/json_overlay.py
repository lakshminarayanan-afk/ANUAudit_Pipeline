import json
import numpy as np
import cv2
from PIL import Image
from config.config import Config
image="/home/htic/mln/pyqt/ml/seg_inp/INDTNCHE0100237_22_Head_Transcerebellar plane_001517_SPSnapshot_27529.png"
original_image = cv2.imread(image)  # BGR image
if original_image is None:
    raise ValueError("Image not found or failed to load")
image_resized = cv2.resize(original_image, (1024, 1024))
output = image_resized.copy()

with open("/home/htic/mln/pyqt/seg/output_contours/segmentation_results.json", "r") as f:
    data = json.load(f)

# contour1=data["INDTNCHE0100237_22_Head_Transcerebellar plane_001517_SPSnapshot_27529.png"]["structure_0"]
# print(f"Contour 1: {contour1}")
# print(f"Number of points in Contour 1: {len(contour1)}")
# print(f"Contour type: {type(contour1)}")
# np_contour1 = np.array(contour1)
# print(f"Contour 1 shape w np:{np_contour1.shape}")
# np_contour2 = np_contour1.transpose(1,0,2)
# print(f"Contour 1 shape after transpose: {np_contour2.shape}")
for c in range(32):
    contour=data["INDTNCHE0100237_22_Head_Transcerebellar plane_001517_SPSnapshot_27529.png"][f"structure_{0}"] 
    np_contour = np.array(contour)
    np_contour2 = np_contour.transpose(1,0,2)

    cv2.drawContours(
                output,
                [np_contour2], 
                -1,
                Config.structure_colors[c],
                Config.CONTOUR_THICKNESS
            )
    
output=cv2.resize(output, (1024, 1024)) 

cv2.imwrite("/home/htic/mln/pyqt/seg/output_contours/overlay_output1.png", output)
# for img_name, contours in data.items():
#     if img_name == "INDTNCHE0100237_22_Head_Transcerebellar plane_001517_SPSnapshot_27529.png":  
#         print(f"Image: {img_name}")
#         for contour in contours:
#             contour_array = np.array(contour)
#             print(contour_array)