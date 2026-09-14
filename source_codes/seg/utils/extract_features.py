import numpy as np
import cv2

IMG_AREA= 1024*1024

def mask_area(mask):
    return mask.sum()/IMG_AREA

def centroid(mask):
    #print(mask.shape)
    ys,xs=np.where(mask>0)
    if len(xs)==0:
        return 0.0,0.0
    return xs.mean()/1024,ys.mean()/1024

def compactness(mask):
    mask=mask.astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return 0.0
    cnt=max(contours,key=cv2.contourArea)
    area=cv2.contourArea(cnt)
    perimeter=cv2.arcLength(cnt,True)
    return (perimeter**2)/(4* np.pi*area+1e-6)

def extract_struct_features(pred_mask):
    
    # shape of pred mask is (class, height,width)
    falx=pred_mask[4]
    csp=pred_mask[10]
    skull_inner=pred_mask[3]
    skull_outer=pred_mask[9]

    falx_present= int(falx.sum()>0)
    csp_present =int(csp.sum()>0)
    skull_present=int((skull_inner.sum()+skull_outer.sum())>0)

    falx_area=mask_area(falx)
    csp_area= mask_area(csp)
    skull_area=mask_area(skull_inner)+ mask_area(skull_outer)

    falx_cx,falx_cy=centroid(falx)
    csp_cx,csp_cy=centroid(csp)

    falx_csp_overlap=(falx & csp).sum()/IMG_AREA

    horn_comp_1=compactness(pred_mask[15]) # anterior horn of lateral ventricle 1
    horn_comp_2=compactness(pred_mask[14])   #anterior horn of lateral ventricle 2

    thalamus1_area= mask_area(pred_mask[7])  #thalamus 1 

    thalamus2_area=mask_area(pred_mask[0])   # thalamus 2

    total_structures=sum(int(pred_mask[c].sum()>0) for c in range(pred_mask.shape[0]))

    
    return {falx_present,falx_area,csp_present,csp_area,skull_present,skull_area,falx_csp_overlap,falx_cx,falx_cy,csp_cx,csp_cy,
                 horn_comp_1,horn_comp_2,thalamus1_area,thalamus2_area,total_structures}