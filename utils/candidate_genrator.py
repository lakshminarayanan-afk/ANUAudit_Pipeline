RANKS = ["1st", "2nd", "3rd"]

import json
from pathlib import Path

def generate_candidates(classification):
    candidates = {}
    for side, predictions in classification.items():
        side_candidates = []
        seen = set()
        for rank in RANKS:
            prediction = predictions.get(rank)
            if not prediction:
                continue
            anatomy = prediction.get("Anatomy")
            if not anatomy:
                continue
            # Remove duplicate anatomy models
            if anatomy in seen:
                continue
            seen.add(anatomy)
            side_candidates.append({
                "anatomy": anatomy,
                "rank": rank,
                "standard_plane": prediction.get("Standard plane"),
                "confidence": prediction.get(
                    "Standard_plane_confidence"
                )
            })
        candidates[side] = side_candidates
    return candidates


def load_pipeline_inputs(json_folder):
    json_folder = Path(json_folder)
    items = []
    for json_path in json_folder.glob("*.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        candidates = generate_candidates(
            data["classification"]
        )
        for side, side_candidates in candidates.items():
            if not side_candidates:
                continue
            items.append({
                "image_path": data["image_path"],
                "json_path": str(json_path),
                "side": side,
                "candidates": side_candidates,
                # Which canddate are we currently trying?
                "current_index": 0,
                "status": "pending",
                "result": None
            })
    return items

