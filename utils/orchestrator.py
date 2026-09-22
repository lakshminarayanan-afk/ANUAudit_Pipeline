from source_codes.model_config import MODEL_CONFIG
from utils.seg_biom_results_json import write_segmentation_result

from pathlib import Path
import json

def build_initial_batches(items):
    batches = {}
    for item in items:
        candidate = item["candidates"][0]
        anatomy = candidate["anatomy"]
        if anatomy not in batches:
            batches[anatomy] = []
        batches[anatomy].append(item)
    return batches

def move_to_next_candidate(item, next_batches):
    item["current_index"] += 1
    index = item["current_index"]
    # No more candidates
    if index >= len(item["candidates"]):
        item["status"] = "unresolved"
        return
    next_candidate = item["candidates"][index]
    next_anatomy = next_candidate["anatomy"]
    if next_anatomy not in next_batches:
        next_batches[next_anatomy] = []
    next_batches[next_anatomy].append(item)

### PIPELINE
# def run_segmentation_pipeline(items):

#     current_batches = build_initial_batches(items)

#     while current_batches:

#         print("\n" + "=" * 70)
#         print("NEW PIPELINE ROUND")
#         print("=" * 70)
#         next_batches = {}

#         for anatomy, batch in current_batches.items():
#             print()
#             print("-" * 70)
#             print(f"Running model: {anatomy}")
#             print(f"Images: {len(batch)}")
#             print("-" * 70)
#             model_function = MODEL_CONFIG[anatomy]["function"]

#             # ---------------------------------------
#             # Run the actual segmentation model
#             # ---------------------------------------
#             results = model_function(batch)
#             # results should correspond to batch
#             for item, result in zip(batch, results):
#                 status = result["status"]

#                 if status == "standard":
#                     item["status"] = "standard"

#                 # ---------------------------------------
#                 # UNKNOWN / NON-STANDARD → NEXT MODEL
#                 # ---------------------------------------
#                 else:
#                     # print(
#                     #     f"[{status.upper()}] "
#                     #     f"{item['image_path']} "
#                     #     f"({item['side']}) "
#                     #     f"→ next candidate"
#                     # )
#                     move_to_next_candidate(
#                         item,
#                         next_batches
#                     )

#         current_batches = next_batches
        
#     return items

def run_segmentation_pipeline(items):

    current_batches = build_initial_batches(items)

    round_num = 1

    while current_batches:

        print("\n" + "=" * 70)
        print(f"PIPELINE ROUND {round_num}")
        print("=" * 70)

        next_batches = {}

        for anatomy, batch in current_batches.items():

            print()
            print("-" * 70)
            print(f"Running model: {anatomy}")
            print(f"Images: {len(batch)}")
            print("-" * 70)

            # =====================================================
            # NO MODEL YET
            # =====================================================

            if anatomy in {"Thorax"}:

                print(
                    "[SKIP MODEL] Spine and Thorax model "
                    "not available -> marking Standard"
                )

                for item in batch:

                    json_entry = {
                        "plane": "Spine and Thorax",
                        "plane_quality": "Standard",
                    }

                    write_segmentation_result(
                        item,
                        json_entry
                    )

                    item["status"] = "standard"

                continue

            # =====================================================
            # ACTUAL MODEL
            # =====================================================

            results = run_model_adapter(anatomy, batch)

            for item, result in zip(batch, results):

                status = result["status"]

                if status == "standard":
                    item["status"] = "standard"

                else:
                    move_to_next_candidate(
                        item,
                        next_batches
                    )

        current_batches = next_batches
        round_num += 1

    return items


def run_model_adapter(anatomy, batch):

    model_function = MODEL_CONFIG[anatomy]["function"]

    # Run development code unchanged
    model_function(batch)

    results = []

    for item in batch:

        json_path = Path(item["json_path"])

        with open(json_path, "r") as f:
            data = json.load(f)

        panel = item["panel"]
        rank = item["candidates"][item["current_index"]]["rank"]

        segmentation = data.get("segmentation", {})

        seg_result = (
            segmentation
            .get(panel, {})
            .get(rank, {})
        )

        plane_quality = seg_result.get("plane_quality")

        if plane_quality == "Standard":
            status = "standard"

        elif plane_quality == "Non-Standard":
            status = "non-standard"

        else:
            status = "unknown"

        results.append({
            "status": status,
            "segmentation": seg_result
        })

    return results