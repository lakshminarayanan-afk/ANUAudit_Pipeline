from source_codes.modality.modality_code import modality_inference    

from utils.candidate_genrator import load_pipeline_inputs
from utils.orchestrator import run_segmentation_pipeline

IMAGE_DIRECTORY = (
    r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/TEST"
)

MODALITY_MODEL_PATH = (
    "/home/htic/MLN/PIPELINE/"
    "ANUAudit_Pipeline/weights/MODALITY/modality_model.pth"
)

MODALITY_OUTPUT = (
    "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/TEMP_OUTPUT_HEAD/MODALITY"
)


# ============================================================
# STEP 1 — MODALITY
# ============================================================

# modality_inference(
#     IMAGE_DIRECTORY=IMAGE_DIRECTORY,
#     MODEL_PATH=MODALITY_MODEL_PATH,
#     OUTPUT_FOLDER_PATH=MODALITY_OUTPUT
# )

# ============================================================
# STEP 2 — READ MODALITY JSONs + GENERATE CANDIDATES
# ============================================================

items = load_pipeline_inputs(
    MODALITY_OUTPUT
)

print(f"\nTotal image/side items: {len(items)}")

# ============================================================
# STEP 3 — SEGMENTATION PIPELINE
# ============================================================

final_results = run_segmentation_pipeline(
    items
)