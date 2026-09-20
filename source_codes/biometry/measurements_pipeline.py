from source_codes.biometry.ac_bpd_fl import biomet_inf
from source_codes.audit.FINAL_PIPELINE.inference import (run_biometry_audit, setup_biometry_validators)
from source_codes.biometry.tt_tc_tv import process_single_image_tc, process_single_image_tv, setup_unet_model
from source_codes.biometry.sdvp import setup_model as setup_sdvp_model, process_single_image_sdvp
from utils.seg_biom_results_json import write_biometry_result
from config import Config
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def measurements_pipeline(model_inputs):

    # ============================================================
    # BPD / AC / FL
    # ============================================================
    print("\n" + "=" * 70)
    print("RUN 1 — BPD / AC / FL")
    print("=" * 70)

    bpd_ac_fl_model_inputs = {
        "bpd": model_inputs.get("bpd", []),
        "abdomen": model_inputs.get("abdomen", []),
        "limbs": model_inputs.get("limbs", []),
    }

    print(f"BPD     : {len(bpd_ac_fl_model_inputs['bpd'])} images")
    print(f"Abdomen : {len(bpd_ac_fl_model_inputs['abdomen'])} images")
    print(f"Limbs   : {len(bpd_ac_fl_model_inputs['limbs'])} images")

    print("\n[RUNNING] BPD / AC / FL...")
    biomet_inf(bpd_ac_fl_model_inputs)
    print("[RUNNING] BPD / AC / FL Audit...")
    validators = setup_biometry_validators(device)
    run_biometry_audit(bpd_ac_fl_model_inputs, validators)
                                                               
    # ============================================================
    # TC → TCD / CM; TV -> 
    # ============================================================

    print("\n" + "=" * 70)
    print("RUN 2 — TC / TCD / CM")
    print("=" * 70)

    model_unet = setup_unet_model(tc_tv_model_ckpt= Config.UNET_CHECKPOINT_PATH)

    tc_inputs = model_inputs.get("tc", [])
    for item in tc_inputs:
        image_path = item["image_path"]
        tcd_result, cm_result, units, tcd_points  = process_single_image_tc(image_path,model_unet)
        json_entry = {
            "TCD": tcd_result,
            "CM": cm_result,
            "units": units,
            "confidence": None,
            "points": tcd_points
        }
        write_biometry_result(
            item,
            json_entry
        )

    tv_inputs = model_inputs.get("tv", [])
    for item in tv_inputs:
        image_path = item["image_path"]
        lv_result, units, lv_points = process_single_image_tv(image_path,model_unet)
        json_entry = {
            "LV": lv_result,
            "units": units,
            "confidence" : None,
            "points": lv_points,
        }
        write_biometry_result(
            item,
            json_entry
        )

    # ============================================================
    # SDVP / LIQUOR
    # ============================================================

    print("\n" + "=" * 70)
    print("RUN 3 — LIQUOR")
    print("=" * 70)

    liquor_inputs = model_inputs.get("liquor", [])

    if liquor_inputs:

        sdvp_encoder, sdvp_model, sdvp_preprocess = setup_sdvp_model(
            Config.SDVP_LIQUOR_CHECKPOINT_PATH
        )

        for item in liquor_inputs:

            image_path = item["image_path"]

            json_entry = process_single_image_sdvp(
                image_path,
                sdvp_encoder,
                sdvp_model,
                sdvp_preprocess
            )

            write_biometry_result(
                item,
                json_entry
            )