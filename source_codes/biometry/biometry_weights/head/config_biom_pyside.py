import torch
import os
from config import Config

class Config_biom:
           
    # PATH CONFIGURATION
    # Base directory for the current project
    BASE_DIR = '/home/htic/mln/pyqt/biometry/weights/head'
    # IN main config
    
    # Original MedSAM base weights (Required to initialize architecture)
    MEDSAM_CHECKPOINT = '/home/htic/mln/pyqt/biometry/weights/head/medsam_vit_b.pth'
    # In main cinfig
    
    # Your specific fine-tuned weights
    BEST_CHECKPOINT = os.path.join(BASE_DIR, 'best_model.pth')
    
    # Input data directory for inference
    DIR_DATA = "/home/intern/DATA/DICOM DATA/U95108_M_MENAKA/20260128_101552"
    OUTPUT_BASE_DIR = 'dicom_filtered_bpd_output'
    RESULT_DIR = os.path.join(OUTPUT_BASE_DIR, 'detected_planes')
    CSV_OUTPUT_PATH = os.path.join(OUTPUT_BASE_DIR, 'final_measurements.csv')

    # MODEL HYPERPARAMETERS
    IMG_SIZE = 1024
    BATCH_SIZE = 1
    NUM_CLASSES = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    THRESHOLD = 0.7
    CONTOUR_THICKNESS = 2

    # ANATOMICAL STRUCTURE INDICES (For BPD/OFD Logic)
    INNER_CALVARIUM = 3
    MIDLINE_FALX = 4
    ANTERIOR_FALX = 6
    OUTER_CALVARIUM = 9
    CSP = 10
    CEREBELLUM = 23

    # STRUCTURE NAMES AND VISUALIZATION
    class_names = [
        'Thalamus 2', 'Hemisphere 1', 'Hemisphere 2', 'Inner Calvarium', 
        'Midline Falx', 'Arrow Sign', 'Anterior Midline falx', 'Thalamus 1', 
        'Lateral Sulcus 1', 'Outer Calvarium', 'Cavum Septum Pellucidum', 
        'Lateral Sulcus 2', 'Hippocampal Gyrus 1', 'hippocampal Gyrus 2', 
        'Anterior Horn of Lateral Ventrilces 2', 'Anterior Horn of Lateral Ventricle 1',
        'Pillars of Fornix', 'Posterior Horn of LV 1', 'Posterior horn of LV2', 
        'Lateral ventricle 2', 'Choroid plexus 1', 'Choroid Plexus 2', 
        'Lv measurement', 'Cerebellum', 'Cerebral Vermis', 
        'Petrous part of temporal bone 1', 'nuchal fold', 
        'petrous part of temporal bone 2', 'cisternae magna', 'dural fold', 
        'cerebral peduncle 1', 'cerebral peduncle 2'
    ]

    structure_colors = {
        0: (255, 0, 0),    1: (0, 0, 255),    2: (0, 255, 255),  3: (255, 140, 0),
        4: (128, 0, 128),  5: (0, 100, 0),    6: (50, 205, 50),  7: (255, 255, 0),
        8: (255, 20, 147), 9: (255, 105, 180), 10: (0, 128, 128), 11: (220, 20, 60),
        12: (139, 69, 19), 13: (238, 130, 238), 14: (0, 255, 0),  15: (255, 215, 0),
        16: (255, 0, 255), 17: (160, 82, 45),  18: (218, 112, 214), 19: (75, 0, 130),
        20: (70, 130, 180), 21: (0, 0, 128),   22: (60, 179, 113), 23: (233, 150, 122),
        24: (255, 127, 80), 25: (244, 164, 96), 26: (255, 0, 255), 27: (154, 205, 50),
        28: (173, 255, 47), 29: (153, 50, 204), 30: (128, 0, 0),   31: (178, 34, 34)
    }


    # PLANE CLASSIFICATION RULES
    PLANE_ID_TO_NAME = {0: "transthalamic", 1: "transventricular", 2: "transcerebellar"}

    AREA_THRESHOLDS = {
        "csp": 0.001,
        "arrow": 0.006,
        "cerebellum": 0.013,
        "lateral_ventricle": 0.012
    }
    
    
    
"""
import torch 

class Config:
    #IMAGE_DIR = '/home/htic/padmini/segmentation/data/images'
    #MASK_DIR = '/home/htic/padmini/segmentation/data/masks_multilayer'
    TT_IMAGES='/home/htic/padmini/segmentation/all_structure_training_data_new/Transthalamic/images'
    TT_MASKS='/home/htic/padmini/segmentation/all_structure_training_data_new/Transthalamic/masks'
    TV_IMAGES='/home/htic/padmini/segmentation/all_structure_training_data_new/Transventricular/images'
    TV_MASKS='/home/htic/padmini/segmentation/all_structure_training_data_new/Transventricular/masks'
    TC_IMAGES='/home/htic/padmini/segmentation/all_structure_training_data_new/Transcerebellar/images'
    TC_MASKS='/home/htic/padmini/segmentation/all_structure_training_data_new/Transcerebellar/masks'
    CHECKPOINT_DIR = './checkpoints_medsam_all_new_hd'
    MEDSAM_CHECKPOINT = '/home/intern/segmentation/Model_Improve/best_model.pth'
    TEST_IMAGES='/home/htic/padmini/segmentation/all_structures_test_data/transthalamic/images'
    TEST_MASKS='/home/htic/padmini/segmentation/all_structures_test_data/transthalamic/masks'
    BEST_CHECKPOINT='/home/intern/segmentation/Model_Improve/best_model.pth'
    OUTPUT_DIR='/home/htic/padmini/segmentation/transthalamic_dfe' 
    
    NON_HEAD_OUTPUT='/home/htic/padmini/segmentation/non-head'
    #CROSS_VALID_CHECKPOINT='./checkpoints_medsam_finetune_crossvalid'
    #BEST_CHECKPOINT_KFOLD='/home/htic/padmini/segmentation/Medsam_finetune/checkpoints_medsam_finetune/best_model_aug.pth'
    #OUTPUT_DIR='/home/htic/padmini/segmentation/Medsam_finetune/Results_aug'
    IMG_SIZE = 1024
    BATCH_SIZE = 1
    NUM_EPOCHS = 20    #60 if fine-tuning from scratch, 30 if resuming from checkpoints
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 32
    # Indices for anatomical structure extraction
    INNER_CALVARIUM = 3
    MIDLINE_FALX = 4
    OUTER_CALVARIUM = 9
    ANTERIOR_FALX = 6

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    THRESHOLD=0.7

    FREEZE_IMAGE_ENCODER = True
    UNFREEZE_AFTER_EPOCH = 6 #30 if fine-tuning from scratch, 10 if resuming from checkpoints
   
    CONTOUR_THICKNESS = 2
    '''
    # Structure colors (for contours)
    structure_colors = {
    0: (255, 0, 0),      # Thalamus 2 - Red
    1: (0, 0, 255),      # Cerebral peduncle 2 - Blue
    2: (0, 255, 0),      # Cerebellum - Green
    3: (255, 140, 0),    # Calvarium (Inner table) - Orange
    4: (128, 0, 128),    # Midline Falx - Purple
    5: (0, 255, 255),    # Cerebral peduncle 1 - Cyan
    6: (255, 0, 255),    # Nuchal fold thickness - Magenta
    7: (255, 255, 0),    # Thalamus 1 - Yellow
    8: (50, 205, 50),    # Cisterna Magna - Lime Green
    9: (255, 105, 180),  # Calvarium (Outer table) - Hot Pink
    10: (0, 128, 128),   # Cavum Septum Pellucidum - Teal
        }
    '''
     # Structure colors (for contours)
    structure_colors = {
    0: (255, 0, 0),      # Thalamus 2 - Red
    1: (0, 0, 255),      # Hemisphere 1 - Blue
    2: (0, 255, 255),     # Hemisphere 2 -cyan
    3: (255, 140, 0),    # Inner Calvarium - Orange
    4: (128, 0, 128),    # Midline Falx - Purple
    5: (0, 100, 0),      # Arrow Sign -darkgreen
    6: (50, 205, 50),    # Anterior Midline falx - lime
    7: (255, 255, 0),    # Thalamus 1 - Yellow
    8: (255, 20, 147),    # Lateral Sulcus 1- deeppink
    9: (255, 105, 180),  # Outer Calvarium- Hot Pink
    10: (0, 128, 128),   # Cavum Septum Pellucidum - Teal
    11 : (220,20,60),    #Lateral Sulcus 2 - crimson
    12: (139,69,19),     #Hippocampal Gyrus 1-saddle brown
    13:(238, 130, 238),  #hippocampal Gyrsu 2 -violet
    14 : (0, 255, 0) ,   #Anterior Horn of Lateral Ventrilces 2-green
    15: (255, 215, 0),  # Anterior Horn of Lateral Ventricle 1 - gold
    16: (255, 0, 255) ,  # Pillars of Fornix - magenta

    17:(160, 82, 45),   #posterior horn of lateral ventricle 1
    18:(218, 112, 214) , #posterior horn of lateral ventricle 2
    19:(75, 0, 130),   #lateral ventricle 2
    20:(70, 130, 180), # choroid plexus 1
    21:(0, 0, 128),    # choroid plexus 2
    22:(60, 179, 113) , #Lv measurement 
    23:(233, 150, 122), # cerebellum
    24:(255, 127, 80),    # cereberal vermis 
    25:(244, 164, 96),  #petrous part of temporal bone 1
    26:(255, 0, 255),   # nuchal fold
    27:(154, 205, 50),   #pertrous part of temporal bone 2
    28:(173, 255, 47) , #cisternae magna
    29:(153, 50, 204), # dural fold
    30:(128, 0, 0),  # cerebral peduncle 1
    31:(178, 34, 34) # cerebral peduncle 2
              }
    
    class_names=['Thalamus 2' ,'Hemisphere 1' ,'Hemisphere 2','Inner Calvarium','Midline Falx' ,'Arrow Sign','Anterior Midline falx','Thalamus 1' , 'Lateral Sulcus 1',
                ' Outer Calvarium','Cavum Septum Pellucidum','Lateral Sulcus 2' ,'Hippocampal Gyrus 1','hippocampal Gyrus 2','Anterior Horn of Lateral Ventrilces 2','Anterior Horn of Lateral Ventricle 1',
                'Pillars of Fornix' ,'Posterior Horn of LV 1','Posterior horn of LV2','Lateral ventricle 2','Choroid plexus 1','Choroid Plexus 2','Lv measurement','Cerebellum',
                'Cerebral Vermis','Petrous part of temporal bone 1','nuchal fold','petrous part of temporal bone 2','cisternae magna','dural fold','cerebral peduncle 1','cerebral peduncle 2']
    
    PLANE_CLASS_MAP={
        0:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        1 :[1,2,3,4,6,9,10,14,15,16,17,18,19,20,21,22],
        2 : [0,1,2,3,4,6,7,9,10,12,13,14,15,16,23,24,25,26,27,28,29,30,31]
    }

    #0-tt ,1-tv,2-tc

    PLANE_RULES= { 
        "transthalamic":{
            "csp":10,
            "arrow": 5
        },
         "transventricular":{
            "csp":10,
            "lateral_ventricle":19
        },
        "transcerebellar":{
            "csp":10,
            "cerebellum":23
        }
       
    }

    AREA_THRESHOLDS= {
        "csp": 0.001,
        "arrow": 0.006,
        "cerebellum": 0.013,
        "lateral_ventricle":0.012
    }

    PLANE_ID_TO_NAME= {
        0:"transthalamic",
        1:"transventricular",
        2:"transcerebellar"
    }"""