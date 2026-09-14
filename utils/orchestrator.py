from source_codes.model_config import MODEL_CONFIG

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
def run_segmentation_pipeline(items):

    current_batches = build_initial_batches(items)

    while current_batches:

        print("\n" + "=" * 70)
        print("NEW PIPELINE ROUND")
        print("=" * 70)
        next_batches = {}

        for anatomy, batch in current_batches.items():
            print()
            print("-" * 70)
            print(f"Running model: {anatomy}")
            print(f"Images: {len(batch)}")
            print("-" * 70)
            model_function = MODEL_CONFIG[anatomy]["function"]

            # ---------------------------------------
            # Run the actual segmentation model
            # ---------------------------------------
            results = model_function(batch)
            # results should correspond to batch
            for item, result in zip(batch, results):
                status = result["status"]

                if status == "standard":
                    item["status"] = "standard"

                # ---------------------------------------
                # UNKNOWN / NON-STANDARD → NEXT MODEL
                # ---------------------------------------
                else:
                    # print(
                    #     f"[{status.upper()}] "
                    #     f"{item['image_path']} "
                    #     f"({item['side']}) "
                    #     f"→ next candidate"
                    # )
                    move_to_next_candidate(
                        item,
                        next_batches
                    )

        current_batches = next_batches
        
    return items