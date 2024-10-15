import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


# Función para calcular el EAR
def calcular_EAR(landmarks, lado='left'):
    if lado == 'left':
        puntos = [33, 160, 158, 133, 153, 144]
    else:
        puntos = [362, 385, 387, 263, 373, 380]

    # Obtener las coordenadas de los puntos de referencia
    coords = []
    for punto in puntos:
        x = landmarks[punto][0]
        y = landmarks[punto][1]
        coords.append((x, y))

    # Calcular las distancias
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])

    EAR = (A + B) / (2.0 * C)
    return EAR


# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Inicializar VideoCapture
cap = cv2.VideoCapture(0)

# Variables para el conteo de parpadeos
contador_parpadeos = 0
parpadeo = False

# Umbral EAR para detectar parpadeo
EAR_UMBRAL = 0.21
# Número de frames consecutivos que el EAR debe estar por debajo del umbral
CONSEC_FRAMES = 3
contador_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Obtener todas las coordenadas de los puntos de referencia
            h, w, _ = frame.shape
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))

            # Calcular EAR para ambos ojos
            EAR_izquierdo = calcular_EAR(landmarks, lado='left')
            EAR_derecho = calcular_EAR(landmarks, lado='right')
            EAR_promedio = (EAR_izquierdo + EAR_derecho) / 2.0

            # Dibujar los puntos de los ojos (opcional)
            # Puedes descomentar estas líneas si deseas ver los puntos de referencia
            # for punto in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
            #     cv2.circle(frame, landmarks[punto], 2, (0, 255, 0), -1)

            # Detectar parpadeo
            if EAR_promedio < EAR_UMBRAL:
                contador_frames += 1
            else:
                if contador_frames >= CONSEC_FRAMES:
                    contador_parpadeos += 1
                contador_frames = 0

            # Mostrar el conteo de parpadeos y el EAR
            cv2.putText(frame, f'Parpadeos: {contador_parpadeos}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'EAR: {EAR_promedio:.2f}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar el frame
    cv2.imshow('Deteccion de Parpadeos', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
