from source_codes.modality.modality_code import modality_inference    

from utils.candidate_genrator import load_pipeline_inputs
from utils.orchestrator import run_segmentation_pipeline

modality_inference(
       IMAGE_DIRECTORY = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/8",
        MODEL_PATH = "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/weights/MODALITY/modality_model.pth",
       OUTPUT_FOLDER_PATH = "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/TEMP_OUTPUT/MODALITY"
    )

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