"""
    Biblioteca que auxiliar, por meio de métodos de maquinas, a identificar 
    coisas especificas, como por exemplo: cachorros, cavalos, vacas, etc.

    
"""
import cv2
import numpy as np

#Defina as classes para o mobile net tentar identificar


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]




# Carrega o mobilenet
def load_mobilenet():
    """Não mude ou renomeie esta função
        Carrega o modelo e os parametros da MobileNet. 
        Retorna a rede carregada.
    """
    proto = "MobileNetSSD_deploy.prototxt.txt" # descreve a arquitetura da rede
    model = "MobileNetSSD_deploy.caffemodel" # contém os pesos da rede em si
    net = cv2.dnn.readNetFromCaffe(proto, model)
    return net



def detect(net, frame, CONFIDENCE, COLORS, CLASSES):
    """
    Recebe:
    net - a rede carregada
    frame - uma imagem colorida BGR
    CONFIDENCE - o grau de confiabilidade mínima da detecção
    COLORS - as cores atribídas a cada classe
    CLASSES - o array de classes
    Devolve: 
    img - a imagem com os objetos encontrados
    resultados - os resultados da detecção no formato
    [(label, score, point0, point1),...]
    """
    img = frame.copy()

    resultados = []
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(img, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            resultados.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))
    return img, resultados



"""
    Para executar esse script em frames, e encontrar centros via mobile_net
"""

net = load_mobilenet()
CONFIDENCE = 0.5
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
frame, resultados = detect(net, rgb, CONFIDENCE, COLORS, CLASSES)

#Adiciona as informações do resultado do detect em uma variavel
info = resultados[0]
        
#Acessa as infos e determina as distancias entre os objetos avaliados
distx = info[2]
disty = info[3]

#Calcula o meio do cachorro
meiox = (disty[0] + distx[0])//2
meioy = (disty[1] + distx[1])//2

