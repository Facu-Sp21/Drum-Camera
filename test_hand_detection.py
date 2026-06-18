import cv2
import mediapipe as mp

# Clases de MediaPipe
BaseOptions = mp.tasks.BaseOptions # Clase para opciones base de los modelos de MediaPipe
HandLandmarker = mp.tasks.vision.HandLandmarker # Clase para el detector de manos de MediaPipe
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions # Clase para opciones específicas del detector de manos de MediaPipe

# Configuración del detector
options = HandLandmarkerOptions( #
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task" # Ruta al modelo de MediaPipe para detección de manos
    ),
    num_hands=2
)

# Crear detector
detector = HandLandmarker.create_from_options(options) # Crear una instancia del detector de manos usando las opciones configuradas

# Abrir webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    # OpenCV usa BGR y MediaPipe espera RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crear imagen para MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detectar manos
    result = detector.detect(mp_image)

    # Mostrar cantidad de manos
    num_hands = len(result.hand_landmarks) #hand_landmarks es una lista de las coordenadas de los puntos clave de cada mano detectada, por lo que su longitud indica el número de manos detectadas

    print(f"Manos detectadas: {num_hands}")

    cv2.imshow("Hand Detection Test", frame) # Mostrar el frame original con OpenCV

    if cv2.waitKey(1) == 27: # Si se presiona la tecla ESC, se sale del loop
        break

cap.release() # Liberar la webcam para que pueda ser usada por otras aplicaciones
cv2.destroyAllWindows() # Cerrar todas las ventanas abiertas por OpenCV