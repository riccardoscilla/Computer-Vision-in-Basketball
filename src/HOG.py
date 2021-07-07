
import cv2
import numpy as np
from utility import non_max_suppression

def HOG(hog,roi,c):

    blue = (255,0,0)
    m_fact = 2

    x,y,w,h = c[0],c[1],c[2],c[3]
    patch = roi[y:h,x:w]

    if patch.shape[1] > 0 and patch.shape[0] > 1:
        width,height = patch.shape[1], patch.shape[0]

        if width<64 or height<128: #if small resize to the minimum
            patch = cv2.resize(patch,(64*m_fact,128*m_fact))
        else:
            patch = cv2.resize(patch,(width*m_fact,height*m_fact))

        (rects, _) = hog.detectMultiScale(patch, 
                                            winStride=(4, 4),
                                            padding=(8, 8), 
                                            scale=1.4)
        # (rects, weights) = hog.detectMultiScale(patch)

        rects = np.array(rects)
        merged_rects = non_max_suppression(rects, overlapThresh=0.3)
        
        for rect in merged_rects:
            x,y,w,h = rect[0], rect[1], rect[2], rect[3]
            cv2.rectangle(patch, (x,y),(x+w, y+h), blue, thickness=3*m_fact)

        # resize to original
        patch = cv2.resize(patch,(width,height))

        return patch,len(merged_rects)
    return patch, 0