import scipy.ndimage as ndi
import cv2
import numpy as np

def largest_connected_component(mask):
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    largest = (labeled == (np.argmax(sizes) + 1))
    return largest.astype(np.uint8)

def fill_holes(mask):
    return ndi.binary_fill_holes(mask).astype(np.uint8)

def smooth_mask(mask, kernel=5):
    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return (mask > 0.5).astype(np.uint8)
def overlap_ratio(a, b, eps=1e-6):
    inter = (a & b).sum()
    return inter / (a.sum() + eps)


def enforce_anatomy_rules(masks):
    """
    masks: (C, H, W)
    """
    MIDLINE = 4
    ANT_MIDLINE = 6
    FORNIX = 16
    CSP = 10
    standard_plane=False
    mf=masks[MIDLINE]>0
    amf=masks[ANT_MIDLINE]>0

    if amf.sum()>0 and mf.sum()>0:
        overlap=overlap_ratio(amf,mf)
        if overlap >0.1:
            masks[ANT_MIDLINE]=0

    mid_present = masks[MIDLINE].sum() > 0
    ant_mid_present = masks[ANT_MIDLINE].sum() > 100.0
    #print(masks[ANT_MIDLINE].sum())
    #print(masks[2].sum())

    if mid_present and (ant_mid_present):
        masks[FORNIX] = 0
        
    else:
        masks[CSP] = 0
        masks[ANT_MIDLINE]=0

    return masks



def postprocess_masks(masks):
    #masks=masks.cpu().numpy()
    #print(masks.shape)
    processed = np.zeros_like(masks)
    

    for c in range(masks.shape[0]):
        m = masks[c]

        if m.sum() == 0:
            continue

        m = largest_connected_component(m)
        m = fill_holes(m)
        m = smooth_mask(m)

        processed[c] = m

    processed = enforce_anatomy_rules(processed)
    #processed=suppress_arrow_sign(processed)
    return processed

def get_skull_mask(masks):
    skull= (masks[3]| masks[9]).astype(np.uint8)
    return skull

def largest_cc(mask):
    num,labels=cv2.connectedComponents(mask)
    if num <=1:
        return mask
    areas=[(labels==i).sum() for i in range(1,num)]
    largest=1+np.argmax(areas)
    return (labels==largest).astype(np.uint8)


def circularity(mask):
    mask=(mask>0).astype(np.uint8)*255
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)==0:
        return 0.0
    cnt=max(cnts,key=cv2.contourArea)
    area=cv2.contourArea(cnt)
    perimeter=cv2.arcLength(cnt,True)
    if perimeter==0:
        return 0.0
    return 4*np.pi*area/(perimeter**2)

def radial_std(mask):
    ys,xs=np.where(mask)
    if len(xs)==0:
        return 1.0
    cx,cy=xs.mean(),ys.mean()
    dists=np.sqrt((xs-cx)**2+(ys-cy)**2)
    return np.std(dists)/(np.mean(dists)+1e-6)

def contour_roughness(mask):
    cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)==0:
        return 10.0
    cnt=max(cnts,key=cv2.contourArea)
    #epsilon=0.01*cv2.arcLength(cnt,True)
    #approx=cv2.approxPolyDP(cnt,epsilon,True)
    peri=cv2.arcLength(cnt,True)
    hull=cv2.convexHull(cnt)
    hull_peri=cv2.arcLength(hull,True)
    if hull_peri==0:
        return 10.0
    return peri/hull_peri
    #return len(cnt)/(len(approx)+1e-6)

def isproper_skull(skull_mask):
    skull_mask=get_skull_mask(skull_mask)
    skull=largest_cc(skull_mask)
    circ=circularity(skull_mask)

    rad=radial_std(skull_mask)
    rough=contour_roughness(skull_mask)
    print(circ)
    print(rad)
    print(rough)
    return (circ>0.75 and rad<0.90 and rough <3.5)