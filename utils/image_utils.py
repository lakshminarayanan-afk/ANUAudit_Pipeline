import cv2
import numpy as np
import pydicom


def load_image(img_path):
    """
    Load a DICOM or regular image and return it as RGB uint8.

    Returns:
        img_rgb: np.ndarray
            Shape: (H, W, 3), dtype: uint8
    """

    if img_path.lower().endswith(".dcm"):

        ds = pydicom.dcmread(img_path)

        image = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            image = (
                image * float(ds.RescaleSlope)
                + float(ds.RescaleIntercept)
            )

        # Handle MONOCHROME1
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = np.max(image) - image

        # Normalize to [0, 255]
        image -= image.min()
        image /= image.max() + 1e-6
        image *= 255.0

        image = image.astype(np.uint8)

        # Convert grayscale → RGB
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:
            # DICOM RGB is already RGB
            img_rgb = image

    else:

        # Regular image
        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {img_path}"
            )

        img_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    return img_rgb