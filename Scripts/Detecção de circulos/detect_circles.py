import numpy as np
import cv2

def detect_cicle(img):

    copia = img.copy()
    preto = copia[:,:,0]
    mask_p = np.zeros_like(preto)
    mask_p[preto<255] = 0

    

    gray =  cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    mg = cv2.GaussianBlur(gray, (15,15), 0)

    cores_r = img[:,:,0]
    mask = np.zeros_like(cores_r)
    mask[cores_r < 5] = 255

    mama = np.uint8(np.where(mask == 0))

 
    y = min(mama[0])
    x = len(mama[1])

    circles = cv2.HoughCircles(mg, cv2.HOUGH_GRADIENT, 1.2, 10, param1= 100, param2= 50, minRadius= 1, maxRadius= 100)
    #if circles is None:
    #print(circles)
    circles = np.uint16(np.around(circles))
   

    for (x,y,r) in circles[0 , :]:
        #print(i[2])
        #Comprimento do circulo
        cv2.circle(mask_p, (x, y), r, (255,255,255), 2)

        #centro do Circulo
        cv2.circle(mask_p, (x, y), 1, (255,255,255), 3)
   
    cimg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #print(circles.shape)
    mg =  cv2.cvtColor(mg, cv2.COLOR_GRAY2BGR)
    mask_p = cv2.cvtColor(mask_p, cv2.COLOR_GRAY2BGR)
    return mg,mask_p,circles.shape[1]