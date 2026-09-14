import torch 

class Config_seg:
    #IMAGE_DIR = '/home/htic/padmini/segmentation/data/images'
    #MASK_DIR = '/home/htic/padmini/segmentation/data/masks_multilayer'
    # TT_IMAGES='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_training_full/transthalamic/images'
    # TT_MASKS='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_training_full/transthalamic/masks'
    # TV_IMAGES='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_training_full/transventricular/images'
    # TV_MASKS='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_training_full/transventricular/masks'
    # TC_IMAGES='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_training_full/transcerebellar/images'
    # TC_MASKS='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_training_full/transcerebellar/masks'
    # CHECKPOINT_DIR = './checkpoints_medsam_full_train'
    # MEDSAM_CHECKPOINT = '/mnt/data/padmini/segmentation/medsam_vit_b.pth'
    # TEST_IMAGES1='/home/htic/Downloads/HEAD PLANE- validation -20260211T050324Z-1-001/HEAD PLANE- validation /TV PLANE/Partially standard plane  '
    # TEST_IMAGES='/home/htic/mln/pyqt/ml/seg_inp'
    # TEST_MASKS='/mnt/data/padmini/segmentation/DATA/All_structures/all_structures_test_data/transcerebellar/masks'
    # BEST_CHECKPOINT='/mnt/data/padmini/segmentation/Medsam_finetune/checkpoints_medsam_all_new/loss1/best_model.pth'
    # OUTPUT_DIR='/home/htic/mln/pyqt/seg/output_final1' 
    
    # NON_HEAD_OUTPUT='/home/htic/padmini/segmentation/non-head'
    #CROSS_VALID_CHECKPOINT='./checkpoints_medsam_finetune_crossvalid'
    #BEST_CHECKPOINT_KFOLD='/home/htic/padmini/segmentation/Medsam_finetune/checkpoints_medsam_finetune/best_model_aug.pth'
    #OUTPUT_DIR='/home/htic/padmini/segmentation/Medsam_finetune/Results_aug'
    IMG_SIZE = 1024
    BATCH_SIZE = 1
    NUM_EPOCHS = 60    #60 if fine-tuning from scratch, 30 if resuming from checkpoints
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 32

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    THRESHOLD=0.6

    FREEZE_IMAGE_ENCODER = True
    UNFREEZE_AFTER_EPOCH = 30 #30 if fine-tuning from scratch, 10 if resuming from checkpoints
   
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
    
    class_names=['Thalamus 2' ,'Hemisphere 1' ,'Hemisphere 2','Inner Calvarium','Midline Falx' ,'Arrow Sign','Anterior Midline Falx','Thalamus 1' , 'Lateral Sulcus 1',
                'Outer Calvarium','Cavum Septum Pellucidum','Lateral Sulcus 2' ,'Hippocampal Gyrus 1','Hippocampal Gyrus 2','Anterior Horn of Lateral Ventricles 2','Anterior Horn of Lateral Ventricles 1',
                'Pillars of Fornix' ,'Posterior Horn of LV 1','Posterior horn of LV 2','Lateral Ventricle 2','Choroid Plexus 1','Choroid Plexus 2','LV Measurement','Cerebellum',
                'Cerebral Vermis','Petrous part of Temporal Bone 1','Nuchal Fold','Petrous part of Temporal Bone 2','Cisternae Magna','Dural Fold','Cerebral Peduncle 1','Cerebral Peduncle 2']
    
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
    STRUCTURE_IDX = {
    "csp": 10,
    "arrow": 5,
    "cerebellum": 23,
    "lateral_ventricle": 19
    }

    PLANE_STRUCTURE_MAP = {
    0: ["csp", "arrow"],              # TT
    1: ["csp", "lateral_ventricle"],  # TV
    2: ["csp", "cerebellum"]          # TC
    }



    


    AREA_PRIORS= {
        "csp": 0.001,
        "arrow": 0.006,
        "cerebellum": 0.013,
        "lateral_ventricle":0.012
    }


    PRESENCE_LOSS = {
        "enabled": True,
        "weight": 0.2,
        "beta": 10.0,
        "learnable": True
    }

    PRESENCE_WARMUP_EPOCHS = 5


    PLANE_ID_TO_NAME= {
        0:"Transthalamic Plane",
        1:"Transventricular Plane",
        2:"Transcerebellar Plane"
    }

    # ---- Standard plane decision thresholds ----
    PLANE_CONF_THRESH   = 0.7
    STRUCT_CONF_THRESH  = 0.6

    # weights for combined confidence
    PLANE_CONF_WEIGHT   = 0.85
    STRUCT_CONF_WEIGHT  = 0.6
    STANDARD_CONF_THRESH = 0.5
    
