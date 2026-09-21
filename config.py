import numpy as np
from typing import Dict, List
from pathlib import Path

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


class Config:

    DEVICE = "cuda"

    ## ML
    # CLASS_MODEL_PATH= r"C:/Users/laksh/ANU/MAIN_WEIGHTS/swa_final_27-classes-DACL+ContrLoss+DeeperClsHead#1.pth"
    CLASS_MODEL_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/CLASS/hierarchicalmodel27 (1).pt"
    COLOR_DOPPLER_MODEL_PATH=r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/COLOR_DOPPLER/color_doppler_model.pth"
    MODALITY_MODEL_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/MODALITY/modality_model.pth"

    ## SEG
    # SEG_MODEL_PATH_MEDSAM= '/home/htic/MLN/MAIN_WEIGHTS/best_model.pth'
    SEG_MODEL_HEAD= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/best_model_unet.pth"
    SEG_MODEL_ABD = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/abdomen_best.pth"
    SEG_MODEL_LIMBS = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/limbs_best.pt"
    SEG_MODEL_FACE = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/face_best.pt"

    ## BIOMETRY
    BI0MET_CSV_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/coefficientsGlobalV3.csv"
    #BPD
    # UNET_CHECKPOINT_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/head/head_unet/best_model_head.pth"
    UNET_CHECKPOINT_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/SEG/best_model_unet.pth"
    #FL
    FEMUR_CHECKPOINT= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/femur/new_fuvai_best.pt"
    # ABD
    ABD_CHECKPOINT= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/abd/best_ac_skin_line.pt"
    # SDVP
    SDVP_LIQUOR_CHECKPOINT_PATH = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/liquor/liquor_best_model.ckpt"
    SDVP_PATH_FETALCLIP_CONFIG = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/source_codes/biometry/model/liquor/FetalCLIP_config.json"
    SDVP_FETALCLIP_WEIGHT = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUTOMATE/liquor/FetalCLIP_weights.pt"

    # AUDIT
    YOLO_AC_BPD= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/yolo_ac_bpd.pt"
    YOLO_FL= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/yolo_fl.pt"
    BPD_MASK= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/mask_bpd.pth"
    AC_MASK= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/mask_ac.pt"
    FL_MASK= r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/MAIN_WEIGHTS/AUDIT/mask_fl.pt"

    ET_PLANES = { "Transverse Spine", "Renal Arteries", "Coronal Kidneys", "RSA", "Both Feet", "Open Hands", "Premaxillary Triangle", "Placenta", "Amniotic Fluid or Liquor", "Umbilical Artery", "Situs", "Cervix" }
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
        "Transcerebellar Plane": 23,
        "Transthalamic Plane": 24,
        "Transventricular Plane": 25,
        "Transverse Kidneys": 26
    }

    DOPPLER_CLASSES = {
        "Pelvis" : "3 Vessel Cord",
        "Thorax" : "3 Vessel Trachea",
        "Thorax" : "Inflows",
        "Abdomen" : "Renal Arteries"
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

    BIOM_MODEL_PLANE_CONFIG = {
        "bpd": ["Transthalamic Plane"],
        "tc" : ["Transcerebellar Plane"],
        "tv" : ["Transventricular Plane"],
        "abdomen": ["Abdominal Circumference"],
        "limbs": ["Femur"],
        "liquor" : ["Amniotic Fluid or Liquor"]
    }


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


class Seg_Config:
    
    class Head:
        mandatory_structures =  {
            "Transthalamic Plane": [
                "thalamus 1", "thalamus 2", "arrow sign",
                "csp (cavum septum pellucidum)", "midline falx",
                "anterior horn of lateral ventricles 2", "inner calvarium" , "outer calvarium"
            ],
            "Transventricular Plane": [
                "lateral ventricle 2", "csp (cavum septum pellucidum)",
                "choroid plexus 2", "posterior horn of lateral ventricle 2",
                "midline falx", "anterior horn of lateral ventricles 2",  "inner calvarium" , "outer calvarium"
            ],
            "Transcerebellar Plane": [
                "cerebellum", "cerebellar vermis", "cisternae magna",
                "csp (cavum septum pellucidum)", "cerebral peduncle 1",
                "cerebral peduncle 2", "anterior horn of lateral ventricles 2",  "inner calvarium" , "outer calvarium"
            ],
        }

        structure_thresholds =  {
            "inner calvarium":                       {"contrast": 30.0, "blur": 25.0, "conf": 0.95},
            "outer calvarium":                       {"contrast": 30.0, "blur": 25.0, "conf": 0.95},
            "csp (cavum septum pellucidum)":         {"contrast": 10.0, "blur": 15.0, "conf": 0.65},
            "thalamus 1":                            {"contrast": 10.0, "blur": 15.0, "conf": 0.50},
            "thalamus 2":                            {"contrast": 10.0, "blur": 15.0, "conf": 0.50},
            "choroid plexus 2":                      {"contrast": 15.0, "blur": 15.0, "conf": 0.50},
            "hemisphere 1":                          {"contrast": 10.0, "blur": 10.0, "conf": 0.50},
            "hemisphere 2":                          {"contrast": 10.0, "blur": 10.0, "conf": 0.50},
            "lateral ventricle 2":                   {"contrast":  5.0, "blur": 10.0, "conf": 0.50},
            "cerebellum":                            {"contrast": 12.0, "blur": 15.0, "conf": 0.50},
            "cerebellar vermis":                     {"contrast": 12.0, "blur": 15.0, "conf": 0.50},
            "cisternae magna":                       {"contrast": 10.0, "blur": 10.0, "conf": 0.50},
            "_default":                              {"contrast": 12.0, "blur": 15.0, "conf": 0.50},
        }

        structure_colours = {
            'thalamus 2':                             (255,   0,   0),
            'hemisphere 1':                           (  0,   0, 255),
            'hemisphere 2':                           (  0, 255, 255),
            'inner calvarium':                        (255, 140,   0),
            'midline falx':                           (128,   0, 128),
            'arrow sign':                             (  0, 100,   0),
            'anterior midline falx':                  ( 50, 205,  50),
            'thalamus 1':                             (255, 255,   0),
            'lateral sulcus 1':                       (255,  20, 147),
            'outer calvarium':                        (255, 105, 180),
            'cavum septum pellucidum':                (  0, 128, 128),
            'csp (cavum septum pellucidum)':          (  0, 128, 128),
            'lateral sulcus 2':                       (220,  20,  60),
            'hippocampal gyrus 1':                    (139,  69,  19),
            'hippocampal gyrus 2':                    (238, 130, 238),
            'anterior horn of lateral ventricles 2':  (  0, 255,   0),
            'anterior horn of lateral ventricles 1':  (255, 215,   0),
            'pillars of fornix':                      (255,   0, 255),
            'posterior horn of lv 1':                 (160,  82,  45),
            'posterior horn of lv 2':                 (218, 112, 214),
            'lateral ventricle 2':                    ( 75,   0, 130),
            'choroid plexus 1':                       ( 70, 130, 180),
            'choroid plexus 2':                       (  0,   0, 128),
            'lv measurement':                         ( 60, 179, 113),
            'cerebellum':                             (233, 150, 122),
            'cerebral vermis':                        (255, 127,  80),
            'cerebellar vermis':                      (255, 127,  80),
            'petrous part of temporal bone 1':        (244, 164,  96),
            'nuchal fold':                            (255,   0, 255),
            'petrous part of temporal bone 2':        (154, 205,  50),
            'cisternae magna':                        (173, 255,  47),
            'dural fold':                             (153,  50, 204),
            'cerebral peduncle 1':                    (128,   0,   0),
            'cerebral peduncle 2':                    (178,  34,  34),
        }

    class Abdomen:
        mandatory_structures = {
            "Abdominal Circumference": [
                "Umbilical Vein",
                "Skin Line",
                "Vertebrae",
                "Stomach",
                "Adrenal",
            ],
            "Cord Insertion": [
                "Cord Insertion",
                "Skin Line",
                "Vertebrae",
            ],
            "Transverse Kidneys": [
                "Kidney Cortex",
                "Renal Pelvis",
                "Vertebrae",
            ],
        }

        plane_anchor_structures = {
            "Abdominal Circumference": ["Skin Line"],
            "Cord Insertion": ["Cord Insertion"],
            "Transverse Kidneys": ["Kidney Cortex"],
        }

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

    class Limbs:
        PLANES: List[str] = ["HU", "RU", "FE", "TF"]
        PLANE_NAMES: Dict[str, str] = {
            "HU": "Humerus",
            "RU": "Radius and Ulna",
            "FE": "Femur",
            "TF": "Tibia and Fibula",
        }
        STRUCTURES: List[str] = [
        "Humerus", "Placenta", "Radius", "Ulna", "Hand", "Femur", "Thigh",
        "Femur Length", "Tibia", "Fibula", "Foot", "Lower Leg",
        ]
        BONE_STRUCTURES: List[str] = ["Humerus", "Radius", "Ulna", "Femur", "Tibia", "Fibula"]
        LABEL_MAP: Dict[int, str] = {
            0: "Background",
            1: "Femur",
            2: "Humerus",
            3: "Radius and Ulna",
            4: "Tibia and Fibula",
        }
        BONE_CLASS_ORDER = ("Femur", "Humerus", "Radius and Ulna", "Tibia and Fibula")
        BONE_LABEL_TO_SEG_CLASS= {"Femur": 1, "Humerus": 2, "Radius and Ulna": 3, "Tibia and Fibula": 4}
        MIN_COMPONENT_THRESHOLDS = {
            "Femur": 750,
            "Humerus": 750,
            "Radius and Ulna": 750,
            "Tibia and Fibula": 500,
        }
        MORPHOLOGY_CONFIG = {
            "Femur": {"kernel_size": 7, "max_hole_area": 1000},
            "Humerus": {"kernel_size": 7, "max_hole_area": 1000},
            "Radius and Ulna": {"kernel_size": 5, "max_hole_area": 700},
            "Tibia and Fibula": {"kernel_size": 7, "max_hole_area": 1000},
        }
        NUM_SEG_CLASSES: int = len(LABEL_MAP)
        NUM_BONE_CLASSES: int = len(LABEL_MAP) - 1
        IGNORE_INDEX: int = 255

        # ---------------------------------------------------------------------
        # Paths - EDIT these two lines for your machine, everything else derives from them.
        # ---------------------------------------------------------------------
        # DATA_ROOT = "/mnt/data4tb/anusha/Limbs_Model_copy"
        # DATASET_DIR = DATA_ROOT / "Limbs_Data"

        # CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
        # LOG_DIR = DATA_ROOT / "logs"

        # ---------------------------------------------------------------------
        # Planes (each plane folder holds images that predominantly show one bone group,
        # but all 4 are pooled together at train time into a single dataset/model).
        # ---------------------------------------------------------------------

        from pathlib import Path

        @staticmethod
        def images_dir(split, plane):
            if split.lower() == "train":
                return Config.DATASET_DIR / plane / "images"
            # Unseen Femur inference
            if plane == "FE":
                return Config.DATA_ROOT / "Online_data" / "FE"

            return Config.DATASET_DIR / split.upper() / plane / "images"

        @staticmethod
        def masks_dir(split, plane):
            if split.lower() == "train":
                return Config.DATASET_DIR / plane / "masks"
            else:
                return Config.DATASET_DIR / split.upper() / plane / "masks"

        @classmethod
        def train_images_dir(cls, plane: str) -> Path:
            return cls.images_dir("TRAIN", plane)

        @classmethod
        def train_masks_dir(cls, plane: str) -> Path:
            return cls.masks_dir("TRAIN", plane)

        @classmethod
        def val_images_dir(cls, plane: str) -> Path:
            return cls.images_dir("VAL", plane)

        @classmethod
        def val_masks_dir(cls, plane: str) -> Path:
            return cls.masks_dir("VAL", plane)

        @classmethod
        def test_images_dir(cls, plane: str) -> Path:
            return cls.images_dir("TEST", plane)

        @classmethod
        def test_masks_dir(cls, plane: str) -> Path:
            return cls.masks_dir("TEST", plane)

        NATIVE_H = 1080
        NATIVE_W = 1920
        PAD_MULTIPLE = 32

        ARTIFACTS: List[str] = [
            "Humerus Shadowing", "Limb Shadowing", "Rib Shadowing", "Other Shadowing",
            "Radius Ulna Shadowing", "Femur Shadowing", "Tibia Fibula Shadowing", "Foot Shadowing",
        ] 

        MASK_FOREGROUND_VALUE = 255
        MASK_THRESHOLD = 127



        BONE_NAME_TO_CLASS: Dict[str, int] = {
            "Femur": 1, "Humerus": 2, "Radius": 3, "Ulna": 3, "Tibia": 4, "Fibula": 4,
        }

        PLANE_TO_BONE_NAMES: Dict[str, List[str]] = {
            "HU": ["Humerus"], "RU": ["Radius", "Ulna"], "FE": ["Femur"], "TF": ["Tibia", "Fibula"],
        }

        STRUCTURE_LABEL_PRIORITY: List[str] = ["Radius", "Ulna", "Humerus", "Femur", "Tibia", "Fibula"]

        BONE_CLASS_TO_SHADOW_NAMES: Dict[int, List[str]] = {
            1: ["Femur Shadowing"], 2: ["Humerus Shadowing"],
            3: ["Radius Ulna Shadowing"], 4: ["Tibia Fibula Shadowing"],
        }

        BACKGROUND_WEIGHT: float = 0.1
        CLASS_WEIGHTS: List[float] = [BACKGROUND_WEIGHT, 0.276, 0.369, 0.3815, 0.3765]

        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------
        IN_CHANNELS: int = 1
        EMBED_DIM: int = 96
        DEPTHS: List[int] = [2, 2, 6, 2]
        NUM_HEADS: List[int] = [3, 6, 12, 24]
        WINDOW_SIZE: int = 7
        FPN_CHANNELS: int = 128
        PROJ_DIM: int = 128            # existing pooled, image-level SupCon projection head output dim

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------
        BATCH_SIZE: int = 1
        NUM_WORKERS: int = 4
        PIN_MEMORY: bool = True
        NUM_EPOCHS: int = 150
        LR: float = 1e-4
        WEIGHT_DECAY: float = 1e-5
        LR_PATIENCE: int = 10
        LR_FACTOR: float = 0.5
        EARLY_STOP: int = 20

        # ---------------------------------------------------------------------
        # Loss weights (segmentation / classification / supervised contrastive)
        # ---------------------------------------------------------------------
        W_SEG: float = 1.0
        W_CLS: float = 0.3
        W_CON: float = 0.5
        DICE_WEIGHT: float = 1.0
        CE_WEIGHT: float = 1.0
        SUPCON_TEMPERATURE: float = 0.07

        RANDOM_SEED: int = 42

        AUG_HFLIP: float = 0.5
        AUG_BRIGHTNESS_CONTRAST: float = 0.5
        AUG_BRIGHTNESS_LIMIT: float = 0.15
        AUG_CONTRAST_LIMIT: float = 0.15
        AUG_GAUSS_NOISE: float = 0.3
        AUG_CLAHE: float = 0.2
        AUG_AFFINE: float = 0.5
        AUG_AFFINE_TRANSLATE: float = 0.05
        AUG_AFFINE_ROTATE_LIMIT: int = 10
        AUG_AFFINE_SCALE: tuple = (0.95, 1.05)
        AUG_AFFINE_SHEAR: int = 5

        LOG_INTERVAL: int = 20

        # =======================================================================
        # STANDALONE CONTOUR-ONLY INFERENCE (infer_contours.py) - unchanged
        # =======================================================================
        INFER_IMAGES_DIR: Dict[str, Path] = {
        "HU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/HU/images"),
        "RU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/RU/images"),
        "FE": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/FE/images"),
        "TF": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/TF/images"),
        }

        INFER_MASKS_DIR: Dict[str, Path] = {
        "HU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/HU/masks"),
        "RU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/RU/masks"),
        "FE": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/FE/masks"),
        "TF": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/TF/masks"),
        }
        # INFER_OUTPUT_DIR: Path = DATA_ROOT / "inference_contours_agreement"

        POSTPROCESS_ENABLED: bool = True
        POSTPROCESS_MORPH_KERNEL: int = 3
        POSTPROCESS_MIN_AREA: int = 40
        POSTPROCESS_KEEP_LARGEST_ONLY: bool = True
        POSTPROCESS_FILL_HOLES: bool = True

        CONTOUR_THICKNESS: int = 2
        CONTOUR_CLASS_COLORS_BGR: Dict[int, tuple] = {
            1: (60, 60, 230), 2: (60, 200, 60), 3: (230, 160, 60), 4: (60, 220, 220),
        }

        # =======================================================================
        # NEW: CCDC-STYLE CONTRASTIVE LEARNING
        # -----------------------------------------------------------------------
        # High-confidence positive mining + hard-negative class mining + confidence/confusion
        # weighted InfoNCE contrastive loss (model.py: CCDCProjectionHead, loss.py: CCDCStyleLoss).
        # This is the ONLY new config block added for the CCDC experiment. Every value below has
        # exactly one source of truth - nothing is hard-coded elsewhere.
        # =======================================================================
        CCDC_PROJ_DIM: int = 32                           # pixel-level embedding channels
        CCDC_POSITIVE_CONFIDENCE_THRESHOLD: float = 0.50    # P(GT class) required to count as a positive
        CCDC_MAX_SAMPLES_PER_CLASS: int = 32               # cap on mined pixels per class, per batch
        CCDC_TEMPERATURE: float = 0.07                      # InfoNCE temperature (cosine similarities)
        CCDC_LAMBDA: float = 0.05                           # weight of L_CCDC in L_total

        @classmethod
        def make_dirs(cls) -> None:
            cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
            cls.INFER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        PALETTE_RGB_UPPER_LIMBS = np.array([
            [255, 165,   0],   # 1  Humerus
            [ 60, 210,  60],   # 2  Radius and Ulna
        ], dtype=np.uint8)

        STRUCTURE_PALETTE_IDX_UPPER_LIMBS = {
            "Humerus": 0,
            "Radius and Ulna": 1,
        }

        PALETTE_RGB_LOWER_LIMBS = np.array([
            [255,  60,  60],   # 0  Femur
            [ 30, 144, 255],   # 3  Tibia and Fibula
        ], dtype=np.uint8)

        STRUCTURE_PALETTE_IDX_LOWER_LIMBS = {
            "Femur": 0,
            "Tibia and Fibula": 1,
        }

    class Face:
        ## Face script is dynamic. Standard planes hardcoded at different places
        # Names and casing have been established correctly for the planes tho
        {}