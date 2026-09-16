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

        image_path = data.get("image_path")
        final_prediction = data.get("final_plane_prediction", {})

        if not image_path:
            continue

        for panel, prediction in final_prediction.items():
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