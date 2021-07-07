import cv2
import numpy as np

class mouseCallbackUserData_t:
    def __init__(self,image,n):
        self.image=image
        self.points=[]
        self.points_counter=0
        self.n=n
        self.done=False

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseCallbackUserData = params
        # mouseCallbackUserData.points = np.append(mouseCallbackUserData.points, [[x,y]], axis=0)
        mouseCallbackUserData.points.append([x,y])
        mouseCallbackUserData.points_counter+=1

        cv2.circle(mouseCallbackUserData.image, (x, y), 5, (100, 100, 255), -1)

        if len(mouseCallbackUserData.points)>1:
            p1 = mouseCallbackUserData.points[mouseCallbackUserData.points_counter-2]
            p2 = mouseCallbackUserData.points[mouseCallbackUserData.points_counter-1]
            cv2.line(mouseCallbackUserData.image, p1, p2, (100, 100, 255), 5)

        cv2.imshow("Image", mouseCallbackUserData.image)

        for p1 in mouseCallbackUserData.points:
            for p2 in mouseCallbackUserData.points:
                if p1!=p2 and abs(p1[0]-p2[0])<20 and abs(p1[1]-p2[1])<20:
                    mouseCallbackUserData.done = True

def selectNpoints(image, n):
    mouseCallbackUserData = mouseCallbackUserData_t(image,n)   
    
    cv2.namedWindow("Image",1)
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image",click_event,mouseCallbackUserData)

    while not mouseCallbackUserData.done:
        cv2.waitKey(50)
    mouseCallbackUserData.points = np.array(mouseCallbackUserData.points)

    print("Done!")
    cv2.destroyWindow("Image")

    return mouseCallbackUserData

if __name__ == "__main__":

    im_src = cv2.imread('images/background.png')

    mouseCallbackUserData = selectNpoints(im_src.copy(),4)

    mask = np.zeros((mouseCallbackUserData.image.shape[0],mouseCallbackUserData.image.shape[1],1))

    cv2.fillPoly(mask,[mouseCallbackUserData.points], (255,255,255), 8)
    
    cv2.imshow("Image", mask)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite("images/mask.png", mask)
    