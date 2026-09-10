from source_codes.classification.inference import inference_fn

results= inference_fn(ROOT_DIR=r"C:\Users\laksh\ANU\12_Full_Image_Datasets\ClinicalPatient_13_dcm", MODEL_PATH=r"C:\Users\laksh\ANU\weights\CLASS\hierarchicalmodel27 (1).pt", OUTPUT_FOLDER_PATH=r"C:\Users\laksh\MLN\ANUAudit_Pipeline\TEMP_OUTPUT\CLASS")

# # def run_all_segmentations(groups):

#     for group_name, images in groups.items():

#         print()
#         print("=" * 60)
#         print(f"Running {group_name} model")
#         print(f"Images: {len(images)}")
#         print("=" * 60)

#         # ------------------------------------------------
#         # Get function
#         # ------------------------------------------------

#         segmentation_function = Model_Config.MODEL_FUNCTIONS[group_name]

#         # ------------------------------------------------
#         # Get checkpoint
#         # ------------------------------------------------

#         checkpoint_path = Model_Config.CHECKPOINT_PATHS[group_name]

#         print(f"Checkpoint: {checkpoint_path}")

#         # ------------------------------------------------
#         # Load model ONCE
#         # ------------------------------------------------

#         model = load_model(checkpoint_path)

#         # ------------------------------------------------
#         # Process ALL images for this anatomy
#         # ------------------------------------------------

#         for image_path, info in images.items():

#             print(f"Processing: {image_path}")

#             image = load_image(image_path)

#             result = segmentation_function(
#                 model,
#                 image
#             )

#             # save_result(
#             #     image_path,
#             #     group_name,
#             #     result
#             # )

#         # ------------------------------------------------
#         # Free model before moving to next anatomy
#         # ------------------------------------------------

#         del model

#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()