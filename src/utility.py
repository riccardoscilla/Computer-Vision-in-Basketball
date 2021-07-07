# import the necessary packages
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def non_max_suppression(boxes, overlapThresh):
    
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

def enlarge_rects(rects,val=10):
    new_rects = []
    for c in rects:
        x,y,w,h = c[0],c[1],c[2],c[3]
        new_rect = [x-val,y-val,w+val,h+val]
        new_rects.append(new_rect)
    return new_rects

def display_player_count(frame,tot_players,mean_players):
    white = (255,255,255)
    half = int(frame.shape[1]/2)
    cv2.putText(frame,'# People: '+str(tot_players),(half-500,90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,white,2)
    return frame,mean_players

def init_tracker(tracker,roi,track_subject):
    if track_subject == "referee":
        # referee
        bbox = (1030-70,99,20,80)
    elif track_subject == "player":
        # player    
        bbox = (589-70, 94, 21, 93)
    else:
        cv2.putText(roi,'Select a ROI and then press SPACE or ENTER button!', (1,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),1)
        bbox = cv2.selectROI(roi, False)
        cv2.destroyWindow("ROI selector")

    print("bbox: ",bbox)
    ok = tracker.init(roi, bbox)
    return tracker

def display_ball_possession(frame,rects,count,Lcount,Rcount,switch):
    half = int(frame.shape[1]/2)
    mean = 0
    red = (0,0,255)
    white = (255,255,255)
    for c in rects:
        x,w = c[0],c[2]
        mean += x + (w-x)/2
        
    if len(rects)>0:
        mean /= len(rects)

    if mean > half:
        Rcount += 1
        if Rcount > 20:
            Lcount = 0
            cv2.arrowedLine(frame, (half-40,80), (half+40,80), 
				red, thickness=9, tipLength = 0.5)

            if not switch or switch==2:
                switch = 1
                count +=1 
				# print(count)
        else:
            cv2.arrowedLine(frame, (half+40,80), (half-40,80), 
				red, thickness=9, tipLength = 0.5)

    else:
        Lcount += 1
        if Lcount > 20:
            Rcount = 0
            cv2.arrowedLine(frame, (half+40,80), (half-40,80), 
				red, thickness=9, tipLength = 0.5)

            if not switch or switch==1:
                switch = 2
                count += 1
				# print(count)
        else:
            cv2.arrowedLine(frame, (half-40,80), (half+40,80), 
				red, thickness=9, tipLength = 0.5)

    cv2.putText(frame,'Changes: '+str(count),(half+60,90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,white,2)
    
    return frame,count,Lcount,Rcount,switch
