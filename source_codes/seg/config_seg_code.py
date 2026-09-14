import numpy as np

class Config:

    ROOT_OUTPUT_DIR="/home/htic/MLN/ANU-Audit/OUTPUT"
    CLASSIFIC_OUTPUT_DIR="/home/htic/MLN/ANU-Audit/OUTPUT/CLASS"
    SEG_OUTPUT_DIR= "/home/htic/MLN/ANU-Audit/OUTPUT/SEG"
    BIOM_OUTPUT_DIR= "/home/htic/MLN/ANU-Audit/OUTPUT/BIOM"
    AUDIT_OUTPUT_DIR= "/home/htic/MLN/ANU-Audit/OUTPUT/BIOM_AUDIT"
    EXPORT_DIR= "/home/htic/DHIVYA"
    AUDIT_JSON= "/home/htic/MLN/ANU-Audit/AUDIT_STATE"

    ## Input folder
    INPUT_FOLDER= "/home/htic/MLN/12_Full_Image_Datasets"

    ## ML
    # CLASS_MODEL_PATH= "/home/htic/MLN/weights/swa_final_27-classes-DACL+ContrLoss+DeeperClsHead#1.pth"
    CLASS_MODEL_PATH = "/home/htic/MLN/ANU-Audit/ml/NEW/hierarchicalmodel27 (1).pt"
    DOPPLER_MODEL_PATH="/home/htic/MLN/weights/best_doppler_model.pth"

    ## SEG
    # SEG_MODEL_PATH_MEDSAM= '/home/htic/MLN/weights/best_model.pth'
    SEG_MODEL_HEAD= "/home/htic/MLN/weights/SEG/best_model_unet.pth"
    SEG_MODEL_ABD = "/home/htic/MLN/weights/SEG/abdomen_best.pth"
    SEG_MODEL_LIMBS = "/home/htic/MLN/weights/SEG/limbs_best.pt"
    SEG_MODEL_FACE = "/home/htic/MLN/weights/SEG/face_best.pt"
    ADAPTER_CHECKPOINT_PATH= "/home/htic/MLN/weights/adpt_head_full_train_best.pth"

    ## BIOMETRY

    # MODE 1
    #BPD
    # MEDSAM_CHECKPOINT = '/home/htic/MLN/weights/medsam_vit_b.pth'
    # BEST_CHECKPOINT = "/home/htic/MLN/weights/best_model_head.pth"
    UNET_CHECKPOINT_PATH = "/home/htic/MLN/weights/AUTOMATE/head/head_unet/best_model_head.pth"
    #FL
    FEMUR_CHECKPOINT= "/home/htic/MLN/weights/AUTOMATE/femur/new_fuvai_best.pt"
    # ABD
    ABD_CHECKPOINT= "/home/htic/MLN/weights/AUTOMATE/abd/best_ac_skin_line.pt"


    # AUDIT
    YOLO_AC_BPD= "/home/htic/MLN/weights/AUDIT/yolo_ac_bpd.pt"
    YOLO_FL= "/home/htic/MLN/weights/AUDIT/yolo_fl.pt"
    BPD_MASK= "/home/htic/MLN/weights/AUDIT/mask_bpd.pth"
    AC_MASK= "/home/htic/MLN/weights/AUDIT/mask_ac.pt"
    FL_MASK= "/home/htic/MLN/weights/AUDIT/mask_fl.pt"

    # MODE 2
    TEMP_SEG= '/home/htic/MLN/ANU-Audit/seg/output_final1'

    # FROM OTHERS
    PATH_FETALCLIP_CONFIG = "/home/htic/MLN/weights/FetalCLIP_config.json"
    PATH_FETALCLIP_WEIGHT = "/home/htic/MLN/weights/FetalCLIP_weights.pt"
    BI0MET_CSV_PATH = "/home/htic/MLN/ANU-Audit/biometry/coefficientsGlobalV3.csv"

    MISC_ANATOMY = "Miscellaneous"
    MISC_PLANE = "Miscellaneous"
    REPETITIVE_ANATOMY = "Repetitive"
    REPETITIVE_PLANE = "Repetitive"
    group_map = {
        # Brain
        'Transcerebellar plane': 'Head',
        'Transthalamic plane': 'Head',
        'Transventricular plane': 'Head',

        # Thorax / Cardiac
        '3 Vessel ViewOrTrachea or PAS': 'Thorax',
        '4 Chamber View of Heart': 'Thorax',
        'LVOT': 'Thorax',
        'RVOT': 'Thorax',

        # Abdomen
        'Abdominal Circumference': 'Abdomen',
        'Cord Insertion': 'Abdomen',
        'Transverse Kidneys': 'Abdomen',
        'Coronal Kidneys': 'Abdomen',

        # Fetal Environment
        'Amniotic Fluid or Liquor': 'Fetal Environment',
        'Placenta': 'Fetal Environment',
        'Cervix': 'Fetal Environment',

        # Lower Limbs
        'Humerus': 'Upper Limbs', 
        'Radius and Ulna': 'Upper Limbs', 
        'Open Hands': 'Upper Limbs',

        # Upper Limbs
        'Femur': 'Lower Limbs', 
        'Tibia and Fibula': 'Lower Limbs', 
        'Both Feet': 'Lower Limbs',

        # Spine
        'Sagittal Spine': 'Spine',
        'Coronal Spine': 'Spine',
        "Full Body Coronal View" : 'Spine',

        # Face profile
        'Median Facial Profile': 'Face',
        'Nose and Mouth': 'Face',
        'Orbits and Lenses': 'Face',
        'Premaxillary Triangle': 'Face'
    }


    PALETTE_RGB = np.array([
        [  0,   0,   0],   # 0  background
        [255,  60,  60],   # 1  Stomach
        [255, 165,   0],   # 2  Umbilical Vein
        [230, 230,   0],   # 3  Vertebrae
        [ 60, 210,  60],   # 4  Aorta
        [  0, 200, 200],   # 5  IVC
        [120,  60, 255],   # 6  Adrenal
        [255,   0, 160],   # 7  Placenta
        [210, 105,  30],   # 8  Gall Bladder
        [  0, 255, 140],   # 9  Kidney Cortex
        [ 30, 144, 255],   # 10 Renal Pelvis
        [255, 215,   0],   # 11 Bladder
        [255,  80, 200],   # 12 Cord Insertion
        [ 50, 255, 255],   # 13 Skin Line

    ], dtype=np.uint8)


    SKIN_PALETTE_IDX = 13


    STRUCTURE_PALETTE_IDX = {
        "Stomach": 1,
        "Umbilical Vein": 2,
        "Vertebrae": 3,
        "Aorta": 4,
        "IVC": 5,
        "Adrenal": 6,
        "Placenta": 7,
        "Gall Bladder": 8,
        "Kidney Cortex": 9,
        "Renal Pelvis": 10,
        "Bladder": 11,
        "Cord Insertion": 12,
        "Skin Line": 13,
    }