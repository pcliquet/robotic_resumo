import cv2

def rect(img,contornos):
    y = []
    x = []

    for i in contornos:
        for j in i:
            for n in j:
                x.append(n[0])
                y.append(n[1])
    #print(min(x))
    new_img = cv2.rectangle(img, (min(x), min(y)), (max(x), max(y)), (0, 255, 255))
    return new_img,x,y
