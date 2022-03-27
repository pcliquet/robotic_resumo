import cv2
import numpy as np


def segmenta_linha(bgr,color1,color2):
    """
        Identifica objetos, como por exemplo linhas,
        e retorna uma mascara em preto e branco.
    """
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, color1, color2)
    kernel = np.ones((5, 5), np.uint8)
    morpho = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return morpho