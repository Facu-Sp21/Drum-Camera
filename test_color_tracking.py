import cv2
import numpy as np
import time
import winsound

# Abrir la webcam
cap = cv2.VideoCapture(0)

# Guardar posición Y anterior para calcular velocidad
previous_y = None

# Línea imaginaria donde se detectarán los golpes
HIT_LINE_Y = 300

# Tiempo del último golpe detectado
last_hit_time = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Convertir BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango de rojo (parte baja)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    # Rango de rojo (parte alta)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Buscar contornos en la máscara
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Dibujar la línea de golpe
    cv2.line(
        frame,
        (0, HIT_LINE_Y),
        (frame.shape[1], HIT_LINE_Y),
        (255, 0, 0),
        2
    )

    # Si se detectan contornos, se toma el más grande
    if contours:
        largest = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest)

        # Ignorar objetos demasiado pequeños
        if area > 100:

            # Obtener círculo mínimo que encierra el contorno
            (x, y), radius = cv2.minEnclosingCircle(largest)

            center = (int(x), int(y))

            # Posición vertical actual
            current_y = int(y)

            # Velocidad por defecto
            velocity = 0

            # Si existe una posición anterior calculamos velocidad
            if previous_y is not None:

                # Positivo = baja
                # Negativo = sube
                velocity = current_y - previous_y

                # Detectar si cruzó la línea hacia abajo
                crossed_down = (
                    previous_y < HIT_LINE_Y and
                    current_y >= HIT_LINE_Y
                )

                current_time = time.time()

                # Detectar golpe
                if (
                    crossed_down and
                    velocity > 25 and
                    current_time - last_hit_time > 0.15
                ):
                    winsound.PlaySound("closedHitHat.wav", winsound.SND_ASYNC)

                    last_hit_time = current_time

            # Guardar posición para el siguiente frame
            previous_y = current_y

            # Dibujar círculo alrededor del objeto detectado
            cv2.circle(
                frame,
                center,
                10,
                (0, 255, 0),
                2
            )

            # Mostrar coordenadas
            cv2.putText(
                frame,
                f"X:{int(x)} Y:{int(y)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Mostrar velocidad
            cv2.putText(
                frame,
                f"Velocidad: {velocity}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Color Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()