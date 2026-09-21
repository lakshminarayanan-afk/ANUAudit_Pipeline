def extract_panel(image, panel):
    h, w = image.shape[:2]
    mid_h = h // 2
    mid_w = w // 2
    if panel == "single":
        return image
    elif panel == "left":
        return image[:, :mid_w]
    elif panel == "right":
        return image[:, mid_w:]
    elif panel == "top_left":
        return image[:mid_h, :mid_w]
    elif panel == "top_right":
        return image[:mid_h, mid_w:]
    elif panel == "bottom_left":
        return image[mid_h:, :mid_w]
    elif panel == "bottom_right":
        return image[mid_h:, mid_w:]
    else:
        raise ValueError(f"Unknown panel: {panel}")

# import numpy as np

# def extract_panel(image, panel):
#     h, w = image.shape[:2]

#     mid_h = h // 2
#     mid_w = w // 2

#     # Black image with EXACT same dimensions as input
#     result = np.zeros_like(image)

#     if panel == "single":
#         result[:] = image

#     elif panel == "left":
#         result[:, :mid_w] = image[:, :mid_w]

#     elif panel == "right":
#         result[:, mid_w:] = image[:, mid_w:]

#     elif panel == "top_left":
#         result[:mid_h, :mid_w] = image[:mid_h, :mid_w]

#     elif panel == "top_right":
#         result[:mid_h, mid_w:] = image[:mid_h, mid_w:]

#     elif panel == "bottom_left":
#         result[mid_h:, :mid_w] = image[mid_h:, :mid_w]

#     elif panel == "bottom_right":
#         result[mid_h:, mid_w:] = image[mid_h:, mid_w:]

#     else:
#         raise ValueError(f"Unknown panel: {panel}")

#     return result