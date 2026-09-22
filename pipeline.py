from source_codes.modality.modality_code import modality_inference    

from utils.candidate_genrator import load_pipeline_inputs
from utils.orchestrator import run_segmentation_pipeline
from utils.get_final_plane import final_plane
from utils.biomet_input_planes import prepare_plane_inputs
from source_codes.biometry.measurements_pipeline import measurements_pipeline

from config import Config

IMAGE_DIRECTORY = (
    r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/TEST"
)


MODALITY_OUTPUT = (
    "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/OUTPUTS"
)


# ============================================================
# STEP 1 — MODALITY
# ============================================================

modality_inference(
    IMAGE_DIRECTORY=IMAGE_DIRECTORY,
    MODEL_PATH=Config.MODALITY_MODEL_PATH,
    OUTPUT_FOLDER_PATH=MODALITY_OUTPUT,
    DOPPLER_MODEL_PATH=Config.COLOR_DOPPLER_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH = Config.CLASS_MODEL_PATH, 
    DEVICE= Config.DEVICE
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

# # # ============================================================
# # # STEP 4 — FINAL_PLANE PIPELINE
# # # ============================================================
 
# final_plane(
#     MODALITY_OUTPUT
# )

# # ============================================================
# # STEP 4 — BIOMET INPUT PIPELINE
# # ============================================================
 
# model_inputs = prepare_plane_inputs(
#     MODALITY_OUTPUT
# )

# for model, inputs in model_inputs.items():
#     print(f"{model}: {len(inputs)} images")

# # ============================================================
# # STEP 4 — MEASUREMENTS PIPELINE
# # ============================================================
 
# measurements_pipeline(
#     model_inputs
# )