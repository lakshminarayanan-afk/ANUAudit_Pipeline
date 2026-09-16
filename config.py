import numpy as np


### ABDOMEN
PALETTE_RGB_ABDOMEN = np.array([
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

STRUCTURE_PALETTE_IDX_ABDOMEN = {
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

### FACE
PALETTE_RGB_FACE = np.array([
[255,  60,  60],   # 0  Forehead
[255, 165,   0],   # 1  Nose
[230, 230,   0],   # 2  Maxilla
[ 60, 210,  60],   # 3  Nasal_Bone
[  0, 200, 200],   # 4  Skinline
[120,  60, 255],   # 5  Lips
[255,   0, 160],   # 6  Mandible
[210, 105,  30],   # 7  Chin
[  0, 255, 140],   # 8  Vomer_bone
[ 30, 144, 255],   # 9  Diencephalon
[255, 215,   0],   # 10 Orbits
[255,  80, 200],   # 11 Lenses
[ 50, 255, 255],   # 12 Upper Lip
[160,  80, 255],   # 13 Lower Lip
], dtype=np.uint8)

CLASS_NAMES_FACE = [
"Forehead", "Nose", "Maxilla", "Nasal_Bone", "Skinline", "Lips", "Mandible",
"Chin", "Vomer_bone", "Diencephalon", "Orbits", "Lenses", "Upper Lip", "Lower Lip",
]

PALETTE_RGB_UPPER_LIMBS = np.array([
    [255, 165,   0],   # 1  Humerus
    [ 60, 210,  60],   # 2  Radius_Ulna
], dtype=np.uint8)

STRUCTURE_PALETTE_IDX_UPPER_LIMBS = {
    "Humerus": 0,
    "Radius and Ulna": 1,
}

PALETTE_RGB_LOWER_LIMBS = np.array([
    [255,  60,  60],   # 0  Femur
    [ 30, 144, 255],   # 3  Tibia_Fibula
], dtype=np.uint8)

STRUCTURE_PALETTE_IDX_LOWER_LIMBS = {
    "Femur": 0,
    "Tibia and Fibula": 1,
}

class Config:

    ROOT_OUTPUT_DIR=r"C:/Users/laksh/ANU/OUTPUT"
    CLASSIFIC_OUTPUT_DIR=r"C:/Users/laksh/ANU/OUTPUT/CLASS"
    SEG_OUTPUT_DIR= r"C:/Users/laksh/ANU/OUTPUT/SEG"
    BIOM_OUTPUT_DIR= r"C:/Users/laksh/ANU/OUTPUT/BIOM"
    AUDIT_OUTPUT_DIR= r"C:/Users/laksh/ANU/OUTPUT/BIOM_AUDIT"
    EXPORT_DIR= "/home/htic/DHIVYA"
    AUDIT_JSON= r"C:/Users/laksh/ANU/ANU-Audit/AUDIT_STATE"

    ## Input folder
    INPUT_FOLDER= r"C:/Users/laksh/ANU/12_Full_Image_Datasets"

    ## ML
    # CLASS_MODEL_PATH= r"C:/Users/laksh/ANU/weights/swa_final_27-classes-DACL+ContrLoss+DeeperClsHead#1.pth"
    CLASS_MODEL_PATH = r"C:/Users/laksh/ANU/weights/CLASS/hierarchicalmodel27 (1).pt"
    DOPPLER_MODEL_PATH=r"C:/Users/laksh/ANU/weights/best_doppler_model.pth"

    ## SEG
    # SEG_MODEL_PATH_MEDSAM= '/home/htic/MLN/weights/best_model.pth'
    SEG_MODEL_HEAD= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/weights/SEG/best_model_unet.pth"
    SEG_MODEL_ABD = r"C:/Users/laksh/ANU/weights/SEG/abdomen_best.pth"
    SEG_MODEL_LIMBS = r"C:/Users/laksh/ANU/weights/SEG/limbs_best.pt"
    SEG_MODEL_FACE = r"C:/Users/laksh/ANU/weights/SEG/face_best.pt"
    ADAPTER_CHECKPOINT_PATH= r"C:/Users/laksh/ANU/weights/adpt_head_full_train_best.pth"

    ## BIOMETRY

    # MODE 1
    #BPD
    UNET_CHECKPOINT_PATH = r"C:/Users/laksh/ANU/weights/AUTOMATE/head/head_unet/best_model_head.pth"
    #FL
    FEMUR_CHECKPOINT= r"C:/Users/laksh/ANU/weights/AUTOMATE/femur/new_fuvai_best.pt"
    # ABD
    ABD_CHECKPOINT= r"C:/Users/laksh/ANU/weights/AUTOMATE/abd/best_ac_skin_line.pt"
    # SDVP
    SDVP_LIQUOR_CHECKPOINT_PATH = r""


    # AUDIT
    YOLO_AC_BPD= r"C:/Users/laksh/ANU/weights/AUDIT/yolo_ac_bpd.pt"
    YOLO_FL= r"C:/Users/laksh/ANU/weights/AUDIT/yolo_fl.pt"
    BPD_MASK= r"C:/Users/laksh/ANU/weights/AUDIT/mask_bpd.pth"
    AC_MASK= r"C:/Users/laksh/ANU/weights/AUDIT/mask_ac.pt"
    FL_MASK= r"C:/Users/laksh/ANU/weights/AUDIT/mask_fl.pt"

    # MODE 2
    TEMP_SEG= '/home/htic/MLN/ANU-Audit/seg/output_final1'

    # FROM OTHERS
    PATH_FETALCLIP_CONFIG = r"C:/Users/laksh/ANU/weights/FetalCLIP_config.json"
    PATH_FETALCLIP_WEIGHT = r"C:/Users/laksh/ANU/weights/FetalCLIP_weights.pt"
    BI0MET_CSV_PATH = r"C:/Users/laksh/ANU/ANU-Audit/biometry/coefficientsGlobalV3.csv"

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

    seg_group_map = {
        "Head" : ['Transcerebellar plane', 'Transthalamic plane', 'Transventricular plane'],
        "Abdomen" : ['Abdominal Circumference', 'Cord Insertion', 'Transverse Kidneys'], 
        "Lower Limbs" : ['Femur' , 'Tibia and Fibula'] ,
        "Upper Limbs" : ['Humerus' , 'Radius and Ulna'],
        "Face" : [ 'Median Facial Profile', 'Nose and Mouth', 'Orbits and Lenses'],
    }

    ANATOMY_PALETTE_MAP = {
        "Abdomen": {
            "palette": PALETTE_RGB_ABDOMEN,
            "structure_idx": STRUCTURE_PALETTE_IDX_ABDOMEN,
        },

        "Face": {
            "palette": PALETTE_RGB_FACE,
            "structure_idx": {
                "Forehead": 0,
                "Nose": 1,
                "Maxilla": 2,
                "Nasal_Bone": 3,
                "Skinline": 4,
                "Lips": 5,
                "Mandible": 6,
                "Chin": 7,
                "Vomer_bone": 8,
                "Diencephalon": 9,
                "Orbits": 10,
                "Lenses": 11,
                "Upper Lip": 12,
                "Lower Lip": 13,
            },
        },

        "Lower Limbs" : {
            "palette": PALETTE_RGB_LOWER_LIMBS,
            "structure_idx": STRUCTURE_PALETTE_IDX_LOWER_LIMBS
        }, 

        "Upper Limbs" : {
            "palette": PALETTE_RGB_UPPER_LIMBS,
            "structure_idx": STRUCTURE_PALETTE_IDX_UPPER_LIMBS
        }
    }

    ## ANONYMIZATION
    ANONYMIZED_IMAGES_DIR=r"C:/Users/laksh/ANU/SHYAM_DEMO/ANONYMIZED_IMAGES"
    ANONYMIZATION_PRESET_DIR=r"C:/Users/laksh/ANU/ANU-Audit/anany_preset"
