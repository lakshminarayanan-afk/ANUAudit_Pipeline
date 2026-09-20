import json
from pathlib import Path
from config import Config

RANKS = ["1st", "2nd", "3rd"]
SKIP_MODALITIES = {
    "pulse_doppler", "colour_doppler"
}
UNAVAILABLE_ANATOMY = {
    "Thorax",
    "Spine"
}

def generate_candidates(data):
    classification = data.get("classification", {})
    modality = data.get("modality", {})
    split = modality.get("split")
    modality_types = modality.get("type")
    candidates = {}

    if split == "single":
        panel_type = modality_types
        if panel_type in SKIP_MODALITIES:
            print(f"[SKIP] Single panel is {panel_type} → no segmentation")
            return {}
        candidates["single"] = build_side_candidates(classification)
        print(f"[CANDIDATES] Single → {candidates['single']}")

    elif split in ["bi-split", "quad-split"]:
        for panel_name, predictions in classification.items():
            panel_type = modality_types.get(panel_name)
            if panel_type in SKIP_MODALITIES:
                print(f"[SKIP] {panel_name} is {panel_type} → no segmentation")
                continue
            panel_candidates = build_side_candidates(predictions)
            if panel_candidates:
                candidates[panel_name] = panel_candidates
                print(f"[CANDIDATES] {panel_name} ({panel_type}) → {panel_candidates}")
            else:
                print(f"[SKIP] {panel_name} has no valid candidates")

    else:
        print(f"[SKIP] Unknown split type: {split}")
        return {}
    return candidates

def build_side_candidates(predictions):
    side_candidates = []
    seen = set()

    # ========================================================
    # FIRST-RANK BLOCKING CONDITIONS
    # If 1st prediction is either:
    #   1. An unavailable anatomy
    #   2. An ET plane
    #
    # then do NOT allow 2nd/3rd ranks to become candidates.
    # ========================================================
    first_prediction = predictions.get("1st")

    if first_prediction:
        first_anatomy = first_prediction.get("Anatomy")
        first_standard_plane = first_prediction.get("Standard plane")

        # ----------------------------------------------------
        # 1. UNAVAILABLE ANATOMY
        # ----------------------------------------------------
        if first_anatomy in UNAVAILABLE_ANATOMY:
            print(
                f"[NO MODEL] 1st rank anatomy={first_anatomy} "
                f"-> no segmentation model available, "
                f"so 2nd/3rd candidates are blocked"
            )
            return []

        # ----------------------------------------------------
        # 2. ET PLANE
        # ----------------------------------------------------
        if first_standard_plane in Config.ET_PLANES:
            print(
                f"[ET PLANE] 1st rank plane={first_standard_plane} "
                f"-> classification result, no segmentation, "
                f"so 2nd/3rd candidates are blocked"
            )
            return []
        
    for rank in RANKS:
        prediction = predictions.get(rank)
        if not prediction:
            print("There is no prediction. SOMME error in the json strcuture/code")
            continue
        anatomy = prediction.get("Anatomy")
        standard_plane = prediction.get("Standard plane")
        # ====================================================
        # ET PLANE
        # Only relevant for 2nd / 3rd here because 1st was
        # already handled above.
        # ====================================================
        if standard_plane in Config.ET_PLANES:
            print(
                f"[ET PLANE] {standard_plane} "
                f"({rank}) -> classification result, "
                f"no segmentation"
            )
            continue
        # ====================================================
        # UNAVAILABLE ANATOMY
        # Only skip this rank if it is 2nd / 3rd.
        # ====================================================
        if anatomy in UNAVAILABLE_ANATOMY:
            print(
                f"[NO MODEL] Anatomy={anatomy} "
                f"({rank}) -> no segmentation model, "
                f"candidate ignored"
            )
            continue
        if not anatomy:
            continue
        if anatomy in seen:
            print(f"[SKIP] Duplicate anatomy: {anatomy}")
            continue
        seen.add(anatomy)
        side_candidates.append({
            "anatomy": anatomy,
            "rank": rank,
            "standard_plane": prediction.get("Standard plane"),
            "confidence": prediction.get("Standard_plane_confidence")
        })
    return side_candidates

def load_pipeline_inputs(json_folder):
    json_folder = Path(json_folder)
    items = []
    json_paths = list(json_folder.glob("*.json"))
    print(f"[LOAD] Found {len(json_paths)} JSON files")
    for json_path in json_paths:
        # print(f"\n[PROCESSING] {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        modality = data.get("modality", {})
        candidates = generate_candidates(data)
        if not candidates:
            # print(f"[SKIP] {json_path.name} → no segmentation candidates")
            continue
        for panel_name, panel_candidates in candidates.items():
            if not panel_candidates:
                # print(f"[SKIP] {json_path.name} / {panel_name} → no candidates")
                continue
            split = modality.get("split")
            modality_types = modality.get("type")
            if split == "single":
                panel_type = modality_types
            else:
                panel_type = modality_types.get(panel_name)
            items.append({
                "image_path": data["image_path"],
                "json_path": str(json_path),
                "panel": panel_name,
                "modality": panel_type,
                "candidates": panel_candidates,
                "current_index": 0,
                "status": "pending",
                "result": None
            })
            # print(f"[ADDED] {json_path.name} | {panel_name} | {panel_type} | First model: {panel_candidates[0]['anatomy']}")
    print(f"\n[DONE] Total pipeline items: {len(items)}")
    return items