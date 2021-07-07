import cv2
import numpy as np
from utility import *
from subtraction import background_subtraction, background_subtraction_full
from HOG import HOG

import argparse
   
def process(output_name,track_subject, bgs, bgsf, fhog):

    cap = cv2.VideoCapture('CV_basket.mp4')

    if output_name:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter("../"+output_name+".mp4",cv2.VideoWriter_fourcc(*'MP4V'), 
                    10, (frame_width,frame_height))

    i = 0
    mean_players = 0
  
    blue = (255,0,0)
    green = (0,255,0)
    red = (0,0,255)
    white = (255,255,255)

    mask = cv2.imread("images/mask.png")

    background = None
    background_full = cv2.imread("images/background.png")
    background_full_mask = cv2.bitwise_and(background_full,mask)[500:750, :]

    # windows size # block size # block stride # cell size # bins
    # hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    tracker = cv2.TrackerCSRT_create()
    track_trace = []

    Lcount,Rcount = 30,30
    switch, count = None, -1

    while cap.isOpened():
        # Reading the video stream
        ret, frame = cap.read()
        if ret:
            timer = cv2.getTickCount()

            rects_bgs = []
            rects_bgf = []

            frame_mask = cv2.bitwise_and(frame,mask)[500:750, :]

            rects_bgs, background = background_subtraction(frame_mask,background,i,show=bgs)
            rects_bgf = background_subtraction_full(frame_mask,background_full_mask,show=bgsf)

            rects = np.array(rects_bgs+rects_bgf)
            
            merged_rects = non_max_suppression(rects, overlapThresh=0.45)

            roi_tracker = frame[500:750, 70:1320].copy()

            # ---- Detection ----
            merged_rects = enlarge_rects(merged_rects,10)
            tot_players = 0

            if fhog:
                for c in merged_rects:
                    x,y,w,h = c[0],c[1],c[2],c[3]
                    patch, n_detect_players = HOG(hog,frame_mask,c)
                    frame_mask[y:h,x:w] = patch
                    tot_players += n_detect_players
            else:
                for c in merged_rects:
                    x,y,w,h = c[0],c[1],c[2],c[3]
                    if w-x > 70 or h-y >100:
                        patch, n_detect_players = HOG(hog,frame_mask,c)
                        frame_mask[y:h,x:w] = patch
                        tot_players += n_detect_players
                    else:
                        cv2.rectangle(frame_mask,(x,y),(w,h),green,3)
                        tot_players += 1

            # ---- Tracking ----
            if i == 0:
                tracker = init_tracker(tracker,roi_tracker,track_subject)
            ok, bbox = tracker.update(roi_tracker)
            if ok:
                # Tracking success
                p1 = (int(bbox[0]+70), int(bbox[1])) # x,y
                p2 = (int(bbox[0] + bbox[2] +70), int(bbox[1] + bbox[3])) # w,h
                cx,cy = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
                track_trace.append((cx,cy))

            it = 0
            for i in range(len(track_trace),0,-1):
                if i > 1 and it<50:
                    it+=1
                    ptt1,ptt2 = track_trace[i-2], track_trace[i-1]
                    cv2.line(frame_mask,ptt1,ptt2,red,3)

            # ---- Display information ----
            frame,mean_players = display_player_count(frame,tot_players,mean_players)
            frame,count,Lcount,Rcount,switch = display_ball_possession(frame,rects,count,Lcount,Rcount,switch)
            
            fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
            cv2.putText(frame,'fps: '+str(fps),(int(frame.shape[1]/2)-700,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,white,2)
            
            # Put back the ROI in original full image and show
            frame_final = frame.copy()
            frame = cv2.bitwise_and(frame,cv2.bitwise_not(mask))
            frame = cv2.bitwise_or(frame[500:750, :],frame_mask)
            frame_final[500:750, :] = frame

            cv2.imshow("frame",frame_final)
            if output_name:
                out.write(frame_final)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        i+=1                          
    
    cap.release()
    if output_name:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Analytics in Basketball')

    parser.add_argument("-o", type=str,
        help="output video name", default=None)

    parser.add_argument("-ts", choices=['player', 'referee', 'None'],
        help="select the subject to track", default='player')

    parser.add_argument("-bgs", type=bool,
        help="show background subtraction image", default=False)
    
    parser.add_argument("-bgsf", type=bool,
        help="show background subtraction full image", default=False)

    parser.add_argument("-fhog", type=bool,
        help="use only HOG Detector", default=False)

    args = parser.parse_args()

    process(args.o,args.ts,args.bgs,args.bgsf,args.fhog)
    