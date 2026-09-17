import os
import json
from pathlib import Path

def write_patient_results(
    json_path,
    patient_name,
    best,
    bpd_val,
    ofd_val,
    ac_val,
    fl_val,
    ci_val,
    efw,
    ga_weeks,
    ga_days,
    units="mm",
):
    """
    Write/update patient-level results in patient_results.json.
    """

    json_path = Path(json_path)

    # patient_results.json will be placed alongside the image JSONs
    results_path = json_path.parent / "patient_results.json"

    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[patient_name] = {
        "measurements": {
            "BPD": bpd_val,
            "OFD": ofd_val,
            "AC": ac_val,
            "FL": fl_val,
            "CI": ci_val,
        },

        "EFW_g": efw,
        "GA_weeks": ga_weeks,
        "GA_days": ga_days,
        "units": units,

        "confidence": {
            "BPD": (
                best["bpd"]["confidence"]
                if best["bpd"]["confidence"] >= 0
                else None
            ),
            "AC": (
                best["ac"]["confidence"]
                if best["ac"]["confidence"] >= 0
                else None
            ),
            "FL": (
                best["fl"]["confidence"]
                if best["fl"]["confidence"] >= 0
                else None
            ),
        },

        "source_images": {
            "BPD": best["bpd"]["image_path"],
            "AC": best["ac"]["image_path"],
            "FL": best["fl"]["image_path"],
        },
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Patient results saved to: {results_path}")