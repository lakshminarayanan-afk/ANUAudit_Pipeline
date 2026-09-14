import json
from pathlib import Path

def write_segmentation_result(item, json_entry):
    json_path = Path(item["json_path"])

    with open(json_path, "r") as f:
        data = json.load(f)

    if "segmentation" not in data:
        data["segmentation"] = {}

    panel = item["panel"]
    rank = item["candidates"][item["current_index"]]["rank"]

    if panel not in data["segmentation"]:
        data["segmentation"][panel] = {}

    data["segmentation"][panel][rank] = json_entry

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
