import os
import cv2

def Crear_Directorio(Directorio):
    if not os.path.exists(Directorio):
        os.makedirs(Directorio)
        return None
    else:
        pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

Contador_Imagen = 0
Letra = "Y"
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    ROI = frame[100:300, 440:600]
    cv2.imshow("ROI", ROI)

    ROI_GS = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI_GS = cv2.resize(ROI_GS, (50, 50), interpolation = cv2.INTER_AREA)
    cv2.imshow("ROI-GS", ROI_GS)

    frame_aux = frame.copy()
    cv2.rectangle(frame_aux, (440, 100), (600, 300), (0, 0, 255), 5)
    cv2.imshow("LiveCamera", frame_aux)

    k = cv2.waitKey(33)
    if k == ord("e"):
        Contador_Imagen += 1
        Directorio  = "./Entrenamiento/" + str(Letra) + "/"
        Crear_Directorio(Directorio)
        cv2.imwrite(Directorio + str(Contador_Imagen) + ".jpg", ROI_GS)
        print("Imagen " + Letra + " " + str(Contador_Imagen) + " guardada en Entrenamiento.")
    if k == ord("v"):
        Contador_Imagen += 1
        Directorio  = "./Validacion/" + str(Letra) + "/"
        Crear_Directorio(Directorio)
        cv2.imwrite(Directorio + str(Contador_Imagen) + ".jpg", ROI_GS)
        print("Imagen " + Letra + " " + str(Contador_Imagen) + " guardada en Validacion.")
    if k == ord("p"):
        Contador_Imagen = 0
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()