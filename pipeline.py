from source_codes.classification.inference import inference_fn

results= inference_fn(ROOT_DIR="/home/htic/MLN/12_Full_Image_Datasets/ClinicalPatient_1_dcm", MODEL_PATH="/home/htic/MLN/PIPELINE/ANUAudit/weights/classification/hierarchicalmodel27.pt", OUTPUT_FOLDER_PATH="/home/htic/MLN/PIPELINE/ANUAudit/outputs")