import json
import cv2
import numpy as np
import pydicom


DICOM_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/ClinicalPatient_13_dcm/IMG_20260220_1_9.dcm"
JSON_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/TEST/input_points.json"
OUTPUT_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/TEST/output_image.png"


def load_dicom(dicom_path):
    ds = pydicom.dcmread(dicom_path)

    # Get pixel data
    image = ds.pixel_array

    # Apply rescale slope/intercept if present
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        image = (
            image.astype(np.float32) * float(ds.RescaleSlope)
            + float(ds.RescaleIntercept)
        )

    # Normalize to 8-bit for OpenCV display
    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Handle MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = 255 - image

    # Convert grayscale to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def draw_polygons(image, json_data):
    polygons = json_data.get("polygons", [])

    for polygon_data in polygons:

        structure = polygon_data.get("structure", "")
        polygon_type = polygon_data.get("type", "")
        points = polygon_data.get("points", [])

        if polygon_type != "polygon" or not points:
            continue

        # Convert JSON points to numpy array
        pts = np.array(
            [
                [round(point["x"]), round(point["y"])]
                for point in points
            ],
            dtype=np.int32
        )

        # Draw polygon
        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Draw polygon points
        for x, y in pts:
            cv2.circle(
                image,
                (x, y),
                radius=3,
                color=(0, 0, 255),
                thickness=-1
            )

        # Structure label
        x, y = pts[0]

        cv2.putText(
            image,
            structure,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    return image


def main():

    # ---------------------------------------------------------
    # 1. Load DICOM
    # ---------------------------------------------------------

    print("Loading DICOM...")

    image = load_dicom(DICOM_PATH)

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")


    # ---------------------------------------------------------
    # 2. Load JSON
    # ---------------------------------------------------------

    print("Loading JSON...")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)


    # ---------------------------------------------------------
    # 3. Draw polygons
    # ---------------------------------------------------------

    result = draw_polygons(image, json_data)


    # ---------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------

    cv2.imwrite(OUTPUT_PATH, result)

    print(f"Saved: {OUTPUT_PATH}")


    # ---------------------------------------------------------
    # 5. Display
    # ---------------------------------------------------------

    # cv2.imshow("DICOM Polygon Viewer", result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()