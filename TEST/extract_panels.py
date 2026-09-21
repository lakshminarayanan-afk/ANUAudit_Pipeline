import pydicom
import cv2
import numpy as np
from pathlib import Path


import pydicom
import cv2
from pathlib import Path


def extract_dicom_split_regions(dicom_path, output_png):
    """
    Extract split-panel coordinates from DICOM Ultrasound Region Sequence,
    overlay the regions and coordinates on the image, and save as PNG.

    Returns:
        list of dictionaries containing panel coordinates.
    """

    dicom_path = Path(dicom_path)
    output_png = Path(output_png)

    ds = pydicom.dcmread(str(dicom_path))

    # ------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------
    image = ds.pixel_array

    if image.ndim == 2:
        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype("uint8")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif image.ndim == 3 and image.shape[-1] == 3:
        image = image.astype("uint8")

    else:
        raise ValueError(
            f"Unsupported pixel array shape: {image.shape}"
        )

    overlay = image.copy()

    # ------------------------------------------------------------
    # Ultrasound Region Sequence
    # (0018,6011)
    # ------------------------------------------------------------
    region_seq_tag = (0x0018, 0x6011)

    if region_seq_tag not in ds:
        print("No Ultrasound Region Sequence found.")
        return []

    regions = []

    # ------------------------------------------------------------
    # Extract every region
    # ------------------------------------------------------------
    for idx, region in enumerate(ds[region_seq_tag].value):

        # Correct DICOM tags
        min_x = region.get((0x0018, 0x6018))
        min_y = region.get((0x0018, 0x601A))
        max_x = region.get((0x0018, 0x601C))
        max_y = region.get((0x0018, 0x601E))

        if any(x is None for x in [min_x, min_y, max_x, max_y]):
            print(f"Region {idx}: coordinates unavailable")
            continue

        x0 = int(min_x.value)
        y0 = int(min_y.value)
        x1 = int(max_x.value)
        y1 = int(max_y.value)

        region_info = {
            "region_index": idx,
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
            "width": x1 - x0,
            "height": y1 - y0,
        }

        regions.append(region_info)

        # --------------------------------------------------------
        # Print
        # --------------------------------------------------------
        print(
            f"Region {idx}: "
            f"({x0}, {y0}) -> ({x1}, {y1})"
        )

        # --------------------------------------------------------
        # Draw rectangle
        # --------------------------------------------------------
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            2
        )

        # --------------------------------------------------------
        # Draw four corner points
        # --------------------------------------------------------
        corners = [
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
        ]

        for x, y in corners:
            cv2.circle(
                overlay,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

        # --------------------------------------------------------
        # Region label
        # --------------------------------------------------------
        label = (
            f"Region {idx} "
            f"({x0},{y0})-({x1},{y1})"
        )

        label_y = max(y0 - 10, 20)

        cv2.putText(
            overlay,
            label,
            (x0 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------------
        # Coordinate labels
        # --------------------------------------------------------
        cv2.putText(
            overlay,
            f"({x0},{y0})",
            (x0 + 5, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            overlay,
            f"({x1},{y1})",
            (max(x1 - 100, x0 + 5), max(y1 - 5, y0 + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    output_png.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_png),
        overlay
    )

    print(f"\nSaved: {output_png}")
    print(f"Total regions: {len(regions)}")

    return regions


regions = extract_dicom_split_regions(
    "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/ClinicalPatient_13_dcm/IMG_20260220_1_9.dcm",
    "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/TEST/image_regions.png"
)