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

    ## ML
    # CLASS_MODEL_PATH= r"C:/Users/laksh/ANU/MAIN_WEIGHTS/swa_final_27-classes-DACL+ContrLoss+DeeperClsHead#1.pth"
    CLASS_MODEL_PATH = r"C:/Users/laksh/ANU/MAIN_WEIGHTS/CLASS/hierarchicalmodel27 (1).pt"
    DOPPLER_MODEL_PATH=r"C:/Users/laksh/ANU/MAIN_WEIGHTS/best_doppler_model.pth"

    ## SEG
    # SEG_MODEL_PATH_MEDSAM= '/home/htic/MLN/MAIN_WEIGHTS/best_model.pth'
    SEG_MODEL_HEAD= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/best_model_unet.pth"
    SEG_MODEL_ABD = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/abdomen_best.pth"
    SEG_MODEL_LIMBS = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/limbs_best.pt"
    SEG_MODEL_FACE = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/face_best.pt"

    ## BIOMETRY

    # MODE 1
    #BPD
    UNET_CHECKPOINT_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/head/head_unet/best_model_head.pth"
    #FL
    FEMUR_CHECKPOINT= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/femur/new_fuvai_best.pt"
    # ABD
    ABD_CHECKPOINT= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/abd/best_ac_skin_line.pt"
    # SDVP
    SDVP_LIQUOR_CHECKPOINT_PATH = r""


    # AUDIT
    YOLO_AC_BPD= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/yolo_ac_bpd.pt"
    YOLO_FL= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/yolo_fl.pt"
    BPD_MASK= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/mask_bpd.pth"
    AC_MASK= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/mask_ac.pt"
    FL_MASK= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/mask_fl.pt"

    # MODE 2
    TEMP_SEG= '/home/htic/MLN/ANU-Audit/seg/output_final1'

    # FROM OTHERS
    PATH_FETALCLIP_CONFIG = r"C:/Users/laksh/ANU/MAIN_WEIGHTS/FetalCLIP_config.json"
    PATH_FETALCLIP_WEIGHT = r"C:/Users/laksh/ANU/MAIN_WEIGHTS/FetalCLIP_weights.pt"
    BI0MET_CSV_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/coefficientsGlobalV3.csv"

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


    ET_PLANES = { "Transverse Spine", "Coronal Kidneys", "RSA", "Both Feet", "Open Hands", "Premaxillary Triangle", "Placenta", "Amniotic Fluid or Liquor", "Umbilical Artery", "Situs", "Cervix" }
    COLOUR_DOPPLER = "colour_doppler"
    PULSE_DOPPLER = "pulse_doppler"

    plane2idx = {
        "3 Vessel View or PAS": 0,
        "4 Chamber View of Heart": 1,
        "Abdominal Circumference": 2,
        "Amniotic Fluid or Liquor": 3, #ET
        "Both Feet": 4, #ET
        "Cervix": 5, #ET
        "Cord Insertion": 6,
        "Coronal Kidneys": 7, #ET
        "Coronal Spine": 8,
        "Femur": 9,
        "Full Body Coronal View": 10,
        "Humerus": 11,
        "LVOT": 12,
        "Median Facial Profile": 13,
        "Nose and Mouth": 14,
        "Open Hands": 15, #ET
        "Orbits and Lenses": 16,
        "Placenta": 17, #ET
        "Premaxillary Triangle": 18, #ET
        "RVOT": 19,
        "Radius and Ulna": 20,
        "Sagittal Spine": 21,
        "Tibia and Fibula": 22,
        "Transcerebellar plane": 23,
        "Transthalamic plane": 24,
        "Transventricular plane": 25,
        "Transverse Kidneys": 26
    }

    anatomy2idx = {
        "Abdomen": 0,
        "Face": 1,
        "Fetal Environment": 2,
        "Head": 3,
        "Lower Limbs": 4,
        "Spine": 5,
        "Thorax": 6,
        "Upper Limbs": 7
    }

    valid_planes_for_anatomy = {
        "0": [2, 6, 7, 26],
        "3": [25, 24, 23],
        "6": [1, 12, 19, 0],
        "7": [11, 20, 15],
        "4": [9, 22, 4],
        "1": [13, 14, 16, 18],
        "5": [21, 8, 10],
        "2": [3, 17, 5]
    }

    LABEL_NAMES = ['b-mode', 'tinted', 'colour_doppler', 'pulse_doppler', 'split_screen_only', 'quadrant_images']

    AUDIT_TERMINOLOGY = {
        "AC": {
            "Top": "TAD NearField",
            "Bottom": "TAD FarField",
            "Left": "Anterio-Posterior point 1",
            "Right": "Anterio-Posterior point 2",
        },

        "FL": {
            "Start": "Diaphysis 1",
            "End": "Diaphysis 2",
        },

        "BPD": {
            "bpd_top": "Parietal bone NearField",
            "bpd_bottom": "Parietal bone FarField",
            "ofd_start": "Occipito-Frontal point 1",
            "ofd_end": "Occipito-Frontal point 2",
        },
    }