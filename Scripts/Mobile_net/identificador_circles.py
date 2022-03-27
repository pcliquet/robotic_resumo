import cv2
from hough_helper import *

"""
    Rede neural que tem o arquivo hough_helper para identificar circulos em telas
"""

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img = rgb.copy()
img_gray = gray.copy()

circles = cv2.HoughCircles(img_gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=12, param1=100, param2=30, minRadius=40, maxRadius=55)
Centro_circulo = circles[0]
Centro_circulo = Centro_circulo[0]
Centro_circulo = (Centro_circulo[0], Centro_circulo[1])

#Função dentro do arquivo hough_helper.py
frame = desenha_circulos(frame, circles)