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
    )

def run_abdomen_model(image_paths):

    return run_inference_abdomen(
        image_paths=image_paths,
        checkpoint=Config.SEG_MODEL_ABD,
        threshold=0.5,
        vis_confidence=0.70,
    )


def run_limbs_model(image_paths):

    return run_inference_limbs(
        checkpoint=Config.SEG_MODEL_LIMBS,
        output_dir=output_dir,
        image_dir=image_paths,
        device_str = "cuda",
        alpha = 0.35,
        # debug = False,
        min_component_thresholds = None,
        morphology_config = None,
        # progress_callback= progress_callback
    )

def run_face_model(image_paths):
    return run_inference_face(
                checkpoint= Config.SEG_MODEL_FACE,
                image_dir=image_paths,
                output_dir=output_dir, 
                masks_dir = None,
                num_vis_images = 10000,
                seed = 42,
                device_str= "cuda",
                progress_callback = progress_callback,
            )

MODEL_CONFIG = {

    "Head": {
        "function": run_head_model,
    },


    # "Abdomen": {
    #     "function": run_abdomen_model,
    # },

    # "Lower Limbs": {
    #     "function": run_limbs_model,
    # },

    # "Upper Limbs": {
    #     "function": run_limbs_model,
    # },

    # "Face": {
    #     "function": run_face_model,
    # },

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


BIOM_MODEL_PLANE_CONFIG = {
    "bpd": [
        "Transthalamic plane",
    ],

    "abdomen": [
        "Abdominal Circumference",
    ],

    "limbs": [
        "Femur",
    ],

    "tc" : ["Transcerebellar plane"],
    "tv" : ["Transventricular plane"],

    "liquor" : ["Amniotic Fluid or Liquor"]

}
