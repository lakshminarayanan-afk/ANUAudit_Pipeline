from source_codes.modality.modality_code import modality_inference
from source_codes.segmentation.head_model import run_head_model


MODEL_CONFIG = {
    "Head": {
        "function": run_head_model,
    },

    "Lower Limbs": {
        "function": run_lower_limbs_model,
    },

    "Spine": {
        "function": run_spine_model,
    },

    "Fetal Environment": {
        "function": run_fetal_environment_model,
    },

    # Add the rest here
}