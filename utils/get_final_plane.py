import json
from pathlib import Path
from config import Config

def final_plane(json_directory):
    """
    Generate final_plane_prediction for every JSON file.

    Rules:

    1. ET PLANES
       - Do NOT use segmentation.
       - Use classification 1st prediction directly.

    2. B-MODE / TINTED
       - If Standard segmentation prediction exists:
             select Standard.
       - Otherwise, if Non-Standard predictions exist:
             select highest mean_structure_confidence.
       - If all available predictions are Unknown:
             use classification 1st prediction.
       - If there is no segmentation result:
             use classification 1st prediction.

    3. COLOUR DOPPLER
       - No classification result.
       - No segmentation result.
       - final_plane_prediction[panel] = from modality

    4. PULSE DOPPLER
       - Skip completely.
       - No final_plane_prediction is written for that panel.

    Supports:
       - single
       - bi-split
       - quad-split
    """

    json_directory = Path(json_directory)

    if not json_directory.exists():
        raise FileNotFoundError(
            f"JSON directory does not exist: {json_directory}"
        )

    json_files = list(json_directory.glob("*.json"))

    print(f"\nFound {len(json_files)} JSON files")

    for json_path in json_files:

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            modality = data.get("modality", {})
            split = modality.get("split")
            modality_type = modality.get("type", {})

            classification = data.get("classification", {})
            segmentation = data.get("segmentation", {})

            # ====================================================
            # DETERMINE PANELS
            # ====================================================

            if split == "single":
                panels = ["single"]

            elif split in ["bi-split", "quad-split"]:
                # For split images, the panel names are the keys
                # present in modality["type"].
                if isinstance(modality_type, dict):
                    panels = list(modality_type.keys())
                else:
                    panels = []

            else:
                print(
                    f"[WARNING] {json_path.name} - "
                    f"Unknown split type: {split}"
                )
                continue

            # ====================================================
            # FINAL PREDICTIONS
            # ====================================================

            final_predictions = {}

            # ====================================================
            # PROCESS EACH PANEL
            # ====================================================

            for panel in panels:

                # ------------------------------------------------
                # Get modality type for this panel
                # ------------------------------------------------

                if split == "single":

                    current_modality_type = modality_type

                    panel_classification = classification

                    panel_segmentation = segmentation.get(
                        "single", {}
                    )

                else:

                    current_modality_type = modality_type.get(
                        panel
                    )

                    panel_classification = classification.get(
                        panel, {}
                    )

                    panel_segmentation = segmentation.get(
                        panel, {}
                    )

                # ------------------------------------------------
                # PULSE DOPPLER
                # ------------------------------------------------

                if current_modality_type == Config.PULSE_DOPPLER:

                    print(
                        f"[SKIP] {json_path.name} | "
                        f"{panel} | pulse_doppler"
                    )

                    continue

                # ------------------------------------------------
                # CLASSIFICATION 1ST
                # ------------------------------------------------

                classification_1st = (
                    panel_classification.get("1st", {})
                )
                # print(classification_1st)

                classification_plane = (
                    classification_1st.get("Standard plane")
                )
                # print(classification_plane)

                classification_confidence = (
                    classification_1st.get(
                        "Standard_plane_confidence"
                    )
                )

                # =================================================
                # 2. COLOUR DOPPLER
                # =================================================

                if current_modality_type == Config.COLOUR_DOPPLER:

                    if classification_plane is not None:

                        final_predictions[panel] = {
                            "plane": classification_plane,
                            "source": "classification",
                            "rank": "1st",
                            "confidence": classification_confidence
                        }

                        print(
                            f"[COLOUR DOPPLER] {json_path.name} | "
                            f"{panel} -> {classification_plane}"
                        )

                    else:

                        print(
                            f"[WARNING] {json_path.name} | "
                            f"{panel} - Colour Doppler prediction missing"
                        )

                    continue

                # =================================================
                # 3. ET PLANE
                # =================================================
                #
                # ET planes NEVER go through segmentation.
                #
                # Classification itself is the final result.
                # =================================================

                if classification_plane in Config.ET_PLANES:
                    # print(f"classification_plane:{classification_plane}")

                    final_predictions[panel] = {
                        "plane": classification_plane,
                        "source": "classification",
                        "rank": "1st",
                        "confidence": classification_confidence
                    }

                    print(
                        f"[ET PLANE] {json_path.name} | "
                        f"{panel} -> "
                        f"{classification_plane} "
                        f"(classification)"
                    )

                    continue
                # =================================================
                # B-MODE
                # =================================================

                if current_modality_type in ["b-mode", "tinted"]:

                    candidates = []

                    # ------------------------------------------------
                    # Collect segmentation predictions
                    # ------------------------------------------------

                    for rank in ["1st", "2nd", "3rd"]:

                        prediction = panel_segmentation.get(
                            rank
                        )

                        if not prediction:
                            continue

                        plane_quality = prediction.get(
                            "plane_quality"
                        )

                        plane = prediction.get("plane")

                        confidence = prediction.get(
                            "mean_structure_confidence"
                        )

                        candidates.append({
                            "rank": rank,
                            "plane": plane,
                            "plane_quality": plane_quality,
                            "confidence": confidence
                        })

                    # ------------------------------------------------
                    # No segmentation predictions
                    # ------------------------------------------------

                    if not candidates:

                        if classification_plane is not None:
                            # print(f"CLASS_PLANE_SEG:{classification_plane}")

                            final_predictions[panel] = {
                                "plane": classification_plane,
                                "source": "classification",
                                "rank": "1st",
                                "confidence": (
                                    classification_confidence
                                )
                            }

                        continue

                    # =================================================
                    # 1. STANDARD
                    # =================================================

                    standard_candidates = [
                        candidate
                        for candidate in candidates
                        if str(
                            candidate["plane_quality"]
                        ).lower() == "standard"
                    ]

                    if standard_candidates:

                        # The first Standard rank wins.
                        # Since candidates are collected in
                        # 1st -> 2nd -> 3rd order, this preserves
                        # the segmentation ranking.

                        selected = standard_candidates[0]

                        final_predictions[panel] = {
                            "plane": selected["plane"],
                            "source": "segmentation",
                            "rank": selected["rank"],
                            "plane_quality": (
                                selected["plane_quality"]
                            ),
                            "confidence": (
                                selected["confidence"]
                            )
                        }

                        continue

                    # =================================================
                    # 2. NON-STANDARD
                    # =================================================

                    non_standard_candidates = [
                        candidate
                        for candidate in candidates
                        if str(
                            candidate["plane_quality"]
                        ).lower()
                        in ["non-standard", "nonstandard"]
                    ]

                    if non_standard_candidates:

                        selected = max(
                            non_standard_candidates,
                            key=lambda candidate: (
                                candidate["confidence"]
                                if candidate["confidence"]
                                is not None
                                else float("-inf")
                            )
                        )

                        final_predictions[panel] = {
                            "plane": selected["plane"],
                            "source": "segmentation",
                            "rank": selected["rank"],
                            "plane_quality": (
                                selected["plane_quality"]
                            ),
                            "confidence": (
                                selected["confidence"]
                            )
                        }

                        continue

                    # =================================================
                    # 3. ALL UNKNOWN
                    # =================================================

                    unknown_candidates = [
                        candidate
                        for candidate in candidates
                        if str(
                            candidate["plane_quality"]
                        ).lower() == "unknown"
                    ]

                    if (
                        len(unknown_candidates)
                        == len(candidates)
                    ):

                        final_predictions[panel] = {
                            "plane": "unknown",
                            "source": "segmentation",
                            "rank": "1st",
                            "plane_quality": "unknown",
                            "confidence": None
                        }

                        continue

                # =================================================
                # UNKNOWN MODALITY
                # =================================================

                print(
                    f"[WARNING] {json_path.name} | "
                    f"{panel} - Unknown modality type: "
                    f"{current_modality_type}"
                )

            # ====================================================
            # WRITE RESULT
            # ====================================================

            if final_predictions:

                data["final_plane_prediction"] = (
                    final_predictions
                )

            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            print(
                f"[DONE] {json_path.name} -> "
                f"{final_predictions}"
            )

        except Exception as e:

            print(
                f"[ERROR] Failed processing "
                f"{json_path.name}: {e}"
            )