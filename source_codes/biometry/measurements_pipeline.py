from source_codes.biometry.ac_bpd_fl import biomet_inf
from source_codes.biometry.tt_tc_tv import process_single_image_tc, process_single_image_tv, setup_unet_model
from source_codes.biometry.sdvp import setup_model as setup_sdvp_model, process_single_image_sdvp
from utils.seg_biom_results_json import write_biometry_result
from config import Config

def measurements_pipeline(model_inputs):

    # ============================================================
    # BPD / AC / FL
    # ============================================================

    bpd_ac_fl_model_inputs = {
        "bpd": model_inputs.get("bpd", []),
        "abdomen": model_inputs.get("abdomen", []),
        "limbs": model_inputs.get("limbs", []),
    }

    biomet_result = biomet_inf(bpd_ac_fl_model_inputs)

    # ============================================================
    # TC → TCD / CM; TV -> 
    # ============================================================
    model_unet = setup_unet_model(tc_tv_model_ckpt= Config.UNET_CHECKPOINT_PATH)

    tc_inputs = model_inputs.get("tc", [])
    for item in tc_inputs:
        image_path = item["image_path"]
        tcd_result, cm_result, units = process_single_image_tc(image_path,model_unet)
        json_entry = {
            "TCD": tcd_result,
            "CM": cm_result,
            "units": units
        }
        write_biometry_result(
            item,
            json_entry
        )

    tv_inputs = model_inputs.get("tv", [])
    for item in tv_inputs:
        image_path = item["image_path"]
        lv_result, units = process_single_image_tv(image_path,model_unet)
        json_entry = {
            "LV": lv_result,
            "units": units
        }
        write_biometry_result(
            item,
            json_entry
        )

    # ============================================================
    # SDVP / LIQUOR
    # ============================================================

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

    return biomet_result