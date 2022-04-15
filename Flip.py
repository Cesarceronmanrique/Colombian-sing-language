import cv2
import os
#from keras.preprocessing.image import load_img

Letra = 'Y'
Directorio = './Validacion/' + Letra
Contador = 253

Datos = os.listdir(Directorio)
for Imagen in Datos:
    Path_imagen = Directorio + '/'+ Imagen
    Original = cv2.imread(Path_imagen)
    Salida = cv2.flip(Original, 1)
    Nombre = Directorio+str(Contador)
    cv2.imwrite(Directorio +"/%i.jpg"%(Contador), Salida)
    Contador += 1

print("Se han generado hasta " + str(Contador) + " imagenes")