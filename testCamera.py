import cv2

cap = cv2.VideoCapture(0)  # agarra la primera camara disponible

while True:   # loop infinito para mostrar la camara
    success, frame = cap.read() # lee un frame de la camara, success es un booleano que indica si se pudo leer el frame, frame es el frame leido

    if not success: # si no se pudo leer el frame,se cierra la camara y se sale del loop
        break

    cv2.imshow("Camera", frame) 

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release() # libera la camara para que pueda ser usada por otras aplicaciones
cv2.destroyAllWindows() # cierra todas las ventanas abiertas por OpenCV