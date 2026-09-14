import numpy as np
import os
import json

def is_missing_csp(f):
    return (
        f["falx_present"] and
        not f["csp_present"] and
        f["falx_area"] > 0.002
    )
    
def is_missing_falx(f):
    return (
        not f["falx_present"] and
        f["skull_present"]
    )
    

def is_thalamic_abnormality(f):
    ratio = min(f["thalamus1_area"], f["thalamus2_area"]) / \
            (max(f["thalamus1_area"], f["thalamus2_area"]) + 1e-6)

    return (
        f["thalamus1_area"] > 0 and
        f["thalamus2_area"] > 0 and
        ratio < 0.6
    )
    
def is_abnormal_anterior_horns(f):
    return (
        f["horn_comp_1"] > 2.5 or
        f["horn_comp_2"] > 2.5
    )
    
def is_csp_falx_overlap(f):
    return f["falx_csp_overlap"] > 0.01

def is_skull_present(f):
    return (
        f["skull_present"] 
        #f["skull_area"] < 0.05 and
        #f["total_structures]
    )
def is_skull_large(f)  :
    return(f["skull_area"]>500)
    
def identify_anomalies(features):
    
    with open('/home/htic/padmini/segmentation/anomaly_information.json' ,'r') as text_out:
        data =json.load(text_out)["rules"]

    matched=[]
    #map json rule ids to condition functions

    condition_map={
        "Missing CSP": is_missing_csp,
        "Missing Falx": is_missing_falx,
        "Abnormal Thalami":is_thalamic_abnormality,
        "Cleaved Anterior horns":is_abnormal_anterior_horns,
        "Absence of skull" : is_skull_present,
        "Enlarged skull"  : is_skull_large

    }
    for rule in data:
        rule_id=rule["id"]
        if rule_id in condition_map and condition_map[rule_id](features):
            matched.append({"text":rule["text"],"anomaly":rule["severity"]})

    if not matched:
        matched.append({"text":" No specific structural anomaly detected","severity":"normal"})

    return matched
