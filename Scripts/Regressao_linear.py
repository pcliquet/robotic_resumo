"""Siga esses passos para efetuar estimativas de retas
    Faça os imports a seguir
"""

import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression, RANSACRegressor

"""
    CV2 - biblioteca para visualização computacional
    Numpy - biblioteca para manipulação multidimensional de matrizes, dados, etc.
    Math  - biblioteca com recursos matematicos
    scikit-learn -  biblioteca de aprendizado de maquina
"""


def crosshair(img,point, size, color):
    """
        Desenha um crosshair centrado em um point.
        point deve ser uma tupla = (sequencia ordenada de dados) (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    x = int(x)
    y = int(y)
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)




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


def estimar_linha_nas_faixas(img, mask):
    """
        Por meio da mascara preta e branca, consegue identificar os contornos para estimas as linhas que compoem a imagem.
        Para estimar as linhas, usa-se a biblioteca Sk-learn e encontra o coeficiente angular e linear. Dessa forma é possivel
        determinar os pontos iniciais e final da reta ([[(x1,y1),(x2,y2)], [(x1,y1),(x2,y2)]]), e desenhar linhas.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    lines = []

    #Encontra contornos
    for contour in contours:

        x_array = contour[:, :, 0] #contornos em x
        y_array = contour[:, :, 1] #contornos em y

        y_array = y_array.reshape(-1, 1) #obrigatório

        #Regressão
        reg = LinearRegression().fit(y_array, x_array)

        #coeficiente angular e linear
        lm = reg.coef_
        o = reg.intercept_ # quando x = 0


        y1 = 0
        x1 = int(lm * y1 + o)
        y2 = img.shape[0]
        x2 = int(lm * y2 + o)

        #estimativas dos pontos inciais e finais
        point_1 = (x1, y1)
        point_2 = (x2, y2)

        #Desenha retas
        cv2.line(img, point_1, point_2, (255, 150, 190), 2)
        lines.append((point_1, point_2))
   
    return lines

def calcular_equacao_das_retas(linhas):
    """
        Recebe uma lista com pontos e por meio dela estima a equação da reta, salvando o coeficiente angular e linear das restas encontradas num Formato: [(m1,h1), (m2,h2)]
    """
    equations = []
    for linha in linhas:
        point_1, point_2 = linha
        x1, y1 = point_1
        x2, y2 = point_2
        m = (x2 - x1) / (y2 - y1)

        equations.append((m, x1))
    print(equations)
    
    return equations

def calcular_ponto_de_fuga(img, equacoes):
    """
       Recebe os coenficientes das retas e retornar o ponto de encontro entre elas. E desenha esse ponto na imagem.
    """
    equation_1, equation_2, equation_3 = equacoes
    m1, h1 = equation_1
    m2, h2 = equation_2
    m3, h3 = equation_3

    y1 = int((h2 - h1) / (m1 - m2))
    x1 = int(h1 + y1 * m1)

    crosshair(img, (x1, y1), 5, (0, 255, 0))

    # y2 = int((h3 - h1) / (m1 - m3))
    # x2 = int(h1 + y2 * m1)

    # crosshair(img, (x2, y2), 5, (0, 0, 255))

    y3 = int((h2 - h3) / (m3 - m2))
    x3 = int(h3 + y3 * m3)

    crosshair(img, (x3, y3), 5, (0, 0, 255))

    return img,(x1, y1)