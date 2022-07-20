import cv2
import mediapipe as mp
import numpy as np
import imutils
import os
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS

#Función para calcular los puntos extremos de la mano
def puntos_maximos(lista_dedos):
    p_de = 0
    p_iz = 1920
    p_ar = 0
    p_ab = 1000
    for lista in lista_dedos:
        if lista[1] > p_de:
            p_de = lista[1]
        if lista[1] < p_iz:
            p_iz = lista[1]
        if lista[2] > p_ar:
            p_ar = lista[2]
        if lista[2] < p_ab:
            p_ab = lista[2]
    return p_iz, p_de, p_ab, p_ar

#Función para ajusar la imagen
def ajustar_imagen(imagen):
    vector_imagen = image.img_to_array(imagen)
    vector_imagen = np.expand_dims(vector_imagen, axis=0)
    vector_imagen /= 255.
    return vector_imagen

#Función para realizar la predicción
def predecir(vector):
    Lista_etiquetas = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']
    porcentaje = max(vector[0])
    indice = vector[0].tolist().index(porcentaje)
    Letra_predecida = Lista_etiquetas[indice]
    return Letra_predecida, porcentaje

modelo = 'Modelo.h5'                #Se carga la ruta del modelo de la red neuronal
pesos_modelo = 'Pesos.h5'           #Se cargan la ruta de los pesos de la red neuronal
cnn = load_model(modelo)            #Se carga el modelo de la red neuronal
cnn.load_weights(pesos_modelo)      #Se cargan los pesos de la red neuronal
cap = cv2.VideoCapture(0)           #Se inicializa la cámara en el dispositivo especificado como argumento
cap.set(3,1920)                     #Ancho de los fotogramas
cap.set(4,1080)                     #Largo de los fotogramas
alto_mano, ancho_mano = 50, 50      #Alto y Ancho para reescalar la mano detectada
Palabra = []

class_manos = mp.solutions.hands    #Libreria mediapipe que es el detector de manos
manos = class_manos.Hands()         #Primer parametro, FALSE para que no haga la deteccion todo el tiempo
                                    #Solo hará detección cuando hay una confianza alta
                                    #Segundo parametro: número máximo de manos default:2
                                    #Tercer parametro: Confianza miníma de detección  50%
                                    #cuarto parametro: Confianza minima de seguimiento  50%
dibujo = mp.solutions.drawing_utils #Dibujar los 21 puntos criticos de la mano

while (1):
    ret, frame = cap.read()                                 #Lectura de la cámara
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=1000, height=1920)  #Lectura de la cámara
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          #Correción de color a RGB
    copia = frame.copy()                                    #Crear una copia para el preprocesamiento
    resultado = manos.process(color)                        #Almacenar ese procesamiento de las manos

    posiciones = []  #Lista para almacenar las coordenadas de los puntos

    #Si detecta alguna mano
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:             #Buscar la mano dentro de la lista de manos del descriptor
            for id, lm in enumerate(mano.landmark):             #Informacion de cada mano encontrada por el id
                alto, ancho, c = frame.shape                    #Extraer el alto y el año del fotograma
                corx, cory = int(lm.x*ancho), int(lm.y*alto)    #Ubicacion de los puntos en pixeles
                posiciones.append([id,corx,cory])               #Agregar el id del punto de la mano y sus respectivas coordenadas

                #dibujo.draw_landmarks(frame, mano, class_manos.HAND_CONNECTIONS)    #Dibujar esas lineas conectando los puntos
                
            if len(posiciones) !=0:
                izquierda, derecha, arriba, abajo = puntos_maximos(posiciones)      #Se llama lafunción para calcular puntos extremos
                cv2.rectangle(frame, (izquierda - 50, arriba - 50), (derecha + 50, abajo + 50),(0, 255, 0), 3)  #Se dibuja un rectángulo sobre la mano
                copia = frame.copy()                                                                #Se realiza una copia de la ventana original
                imagen_mano = copia[arriba - 50:abajo + 50, izquierda - 50:derecha + 50]            #Se extrae la región que posee la mano
                imagen_mano = cv2.cvtColor(imagen_mano, cv2.COLOR_BGR2GRAY)
                try:
                    imagen_mano = cv2.resize(imagen_mano, (50, 50), interpolation = cv2.INTER_AREA) #Se reescala la imagen para el tamaño con el que se entrenó la red neuronal
                    #cv2.imshow("Videogris",imagen_mano)
                    resultado_mano = ajustar_imagen(imagen_mano)
                    vector = cnn.predict(resultado_mano)    #Se predice con la imagen en array
                    Letra, Porcentaje = predecir(vector)    #Se obtiene la letra de la predicción
                    Texto = str(Letra) + " " + str(int(Porcentaje*100))
                    cv2.putText(frame, Texto, (izquierda - 50, arriba - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                except:
                    cv2.putText(frame, "No detecta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    if k == ord("a"):
        Palabra.append(Letra)
        print("Se añadió la letra: ", Letra)
    if k == ord("h"):
        Texto = "".join(Palabra)
        Voz = gTTS(text = Texto, lang = 'es-us', slow = False)
        Voz.save("Salida.mp3")

cap.release()
cv2.destroyAllWindows()