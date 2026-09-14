from inference_ui_changed import run_inference

if __name__ == "__main__":
        overlay_imgs_dir, csv_results_path, structure_csv_path = main(
                test_img_dir_from_ui="/home/htic/mln/pyqt/biometry/biomet_code_vas_0_ops/2026_in_folder",
                output_dir='/home/htic/mln/pyqt/biometry/biomet_code_vas_0_ops/2026_out_1',
                BEST_CHECKPOINT='/mnt/data/padmini/segmentation/Medsam_finetune/checkpoints_medsam_all_new/loss1/best_model.pth'
            )
