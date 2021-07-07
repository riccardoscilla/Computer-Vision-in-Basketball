import cv2
import numpy as np

def bg_update(frame_gray,bg):
    alfa = 0.3
    bg = np.uint8(bg*(1-alfa) + alfa*frame_gray)
    #bg = frame_gray
    return bg
    
def background_subtraction(frame,background,i,show=False):
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = []
    if i==0:
        background = frame_gray
    else:
        diff = cv2.absdiff(background, frame_gray)

        _, motion_mask = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
            
        background = bg_update(frame_gray,background)

        kernel_dil = np.ones((10,10),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.dilate(motion_mask,kernel_dil)

        contours,_ = cv2.findContours(motion_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if h>=(1.)*w and w>30 and h>=35:
                rects.append([x,y,x+w,y+h])
        
        if show:
            cv2.imshow("background subtraction",motion_mask)

    return rects,background

def background_subtraction_full(frame,background_full,show=False):

    mask2 = cv2.imread("images/mask2.png")[500:750, :]
    mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    background_full_gray = cv2.cvtColor(background_full,cv2.COLOR_BGR2GRAY)
    
    rects = []
    diff = cv2.absdiff(background_full_gray, frame_gray)

    _, motion_mask = cv2.threshold(diff,70,255,cv2.THRESH_BINARY)

    motion_mask = cv2.bitwise_and(motion_mask,cv2.bitwise_not(mask2))

    kernel_dil = np.ones((10,10),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask,kernel_dil)
    
    contours,_ = cv2.findContours(motion_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h>=(1.)*w and w>30 and h>=35:
            rects.append([x,y,x+w,y+h])
    
    if show:
        cv2.imshow("background subtraction full",motion_mask)

    return rects
