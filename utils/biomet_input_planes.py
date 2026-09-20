import json
from pathlib import Path

from config import Config


def prepare_plane_inputs(json_directory, model_plane_config=Config.BIOM_MODEL_PLANE_CONFIG):
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

            if not plane:
                continue

            if str(plane).lower() == "colour_doppler":
                print(
                    f"[SKIP] {json_path.name} | "
                    f"panel={panel} | "
                    f"plane=colour_doppler"
                )
                continue

            plane_quality = prediction.get("plane_quality")

            # ============================================================
            # AMNIOTIC FLUID / LIQUOR EXCEPTION
            # Always add irrespective of plane_quality | Have to remove after egtting Liquor segmenattion model
            # ============================================================

            if str(plane).strip().lower() == "amniotic fluid or liquor":
                print(
                    f"[ADD] {json_path.name} | "
                    f"panel={panel} | "
                    f"plane={plane} | "
                    f"plane_quality={plane_quality}"
                )

                for model_name, planes in model_plane_config.items():
                    if plane in planes:
                        model_inputs[model_name].append({
                            "image_path": image_path,
                            "json_path": str(json_path),
                            "panel": panel,
                            "plane": plane
                        })

                continue

            if plane_quality != "Standard":
                print(
                    f"[SKIP] {json_path.name} | "
                    f"panel={panel} | "
                    f"plane_quality={plane_quality}"
                )
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