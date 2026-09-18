import json
from pathlib import Path

from source_codes.model_config import BIOM_MODEL_PLANE_CONFIG


def prepare_plane_inputs(json_directory, model_plane_config=BIOM_MODEL_PLANE_CONFIG):
    json_directory = Path(json_directory)

    model_inputs = {
        model: []
        for model in model_plane_config
    }

    for json_path in json_directory.glob("*.json"):
        with open(json_path, "r") as f:
            data = json.load(f)

        # ============================================================
        # SKIP BISPLIT / QUAD IMAGES
        # ============================================================

        modality = data.get("modality", {})
        split_type = modality.get("split", "single")

        if split_type in {"bisplit", "quad"}:
            print(
                f"[SKIP] {json_path.name} | "
                f"split={split_type}"
            )
            continue

        # ============================================================
        # IMAGE PATH
        # ============================================================
        
        image_path = data.get("image_path")
        final_prediction = data.get("final_plane_prediction", {})

        if not image_path:
            continue

        for panel, prediction in final_prediction.items():

            if prediction is None:
                print(
                    f"[SKIP] {json_path.name} | "
                    f"panel={panel} | "
                    f"prediction=None"
                )
                continue
            
            plane = prediction.get("plane")
            if str(plane).lower() == "unknown":
                print(
                    f"[SKIP] {json_path.name} | "
                    f"panel={panel} | "
                    f"plane=Unknown"
                )
                continue

            if str(plane).lower() == "colour_doppler":
                print(
                    f"[SKIP] {json_path.name} | "
                    f"panel={panel} | "
                    f"plane=colour_doppler"
                )
                continue

            plane_quality = prediction.get("plane_quality")
            if plane_quality != "Standard":
                print(
                    f"[SKIP] {json_path.name} | "
                    f"panel={panel} | "
                    f"plane_quality={plane_quality}"
                )
                continue

            plane = prediction.get("plane")
            if not plane:
                continue

            for model_name, planes in model_plane_config.items():
                if plane in planes:
                    model_inputs[model_name].append({
                        "image_path": image_path,
                        "json_path": str(json_path),
                        "panel": panel,
                        "plane": plane
                    })

    return model_inputs