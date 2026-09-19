from source_codes.modality.modality_code import modality_inference

from source_codes.seg.UNET_SoftMAk.inf import run_inference_head
from source_codes.seg.ABDOMEN.abdomen_inference import run_inference_abdomen
from source_codes.seg.LIMBS.limbs_inference import run_inference_limbs
from source_codes.seg.FACE.face_inference import run_inference_face

from config import Config

def run_head_model(items):

    return run_inference_head(
        items=items,
        checkpoint=Config.SEG_MODEL_HEAD,
        conf_thresh=0.60,
        min_pixels=100,
        calv_threshold=0.85,
        calv_min_px=1000,
        device_str= Config.DEVICE
    )

def run_abdomen_model(items):

    return run_inference_abdomen(        
        checkpoint=Config.SEG_MODEL_ABD,
        items=items,
        threshold=0.5,
        vis_confidence=0.70,
        device_str= Config.DEVICE,
    )

def run_limbs_model(items):

    return run_inference_limbs(
        checkpoint=Config.SEG_MODEL_LIMBS,
        items=items,
        device_str = Config.DEVICE,
        alpha = 0.35,
        # debug = False,
        min_component_thresholds = None,
        morphology_config = None,
        # progress_callback= progress_callback
    )

def run_face_model(items):
    return run_inference_face(
                checkpoint= Config.SEG_MODEL_FACE,
                items=items,
                masks_dir = None,
                seed = 42,
                device_str= Config.DEVICE,
            )

MODEL_CONFIG = {

    "Head": {
        "function": run_head_model,
    },

    "Abdomen": {
        "function": run_abdomen_model,
    },

    "Lower Limbs": {
        "function": run_limbs_model,
    },

    "Upper Limbs": {
        "function": run_limbs_model,
    },

    "Face": {
        "function": run_face_model,
    },

    # "Spine": {
    #     "function": run_spine_model,
    # },

    # "Thorax": {
    #     "function": run_spine_model,
    # },

    # "Fetal Environment": {
    #     "function": run_fetal_environment_model,
    # }
}
