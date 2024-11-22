import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
import math

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

def estimar_pose_cabeza(landmarks, frame_shape):
    """
    Estima la pose de la cabeza y devuelve los ángulos de pitch, yaw y roll.
    """
    # Definir puntos de referencia 3D en el modelo (en mm)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
        (0.0, -330.0, -65.0),        # Chin (Landmark 152)
        (-225.0, 170.0, -135.0),     # Left eye left corner (Landmark 33)
        (225.0, 170.0, -135.0),      # Right eye right corner (Landmark 263)
        (-150.0, -150.0, -125.0),    # Left Mouth corner (Landmark 61)
        (150.0, -150.0, -125.0)      # Right mouth corner (Landmark 291)
    ], dtype=np.float64)

    # Definir puntos de referencia 2D en la imagen
    image_points = np.array([
        landmarks[1],     # Nose tip
        landmarks[152],   # Chin
        landmarks[33],    # Left eye left corner
        landmarks[263],   # Right eye right corner
        landmarks[61],    # Left Mouth corner
        landmarks[291]    # Right mouth corner
    ], dtype=np.float64)

    # Obtener dimensiones de la imagen
    size = frame_shape

    # Establecer la cámara interna (focal length y centro de la imagen)
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Asumir que no hay distorsión
    dist_coeffs = np.zeros((4, 1))  # Distorsión cero

    # Resolver PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    # Convertir el vector de rotación a matriz de rotación
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calcular ángulos de Euler
    # Pitch, Yaw, Roll
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
        roll = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
        roll = 0

    return pitch, yaw, roll

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

# Umbral EAR para detectar parpadeo
EAR_UMBRAL = 0.21
# Número de frames consecutivos que el EAR debe estar por debajo del umbral para considerar un parpadeo
CONSEC_FRAMES_PARPADEO = 3
contador_frames_parpadeo = 0

# Variables para el conteo de tiempo con ojos cerrados
total_tiempo_cerrado = 0.0
tiempo_inicio_cierre = None

# Definir el tiempo mínimo (en segundos) para considerar que los ojos están cerrados y no un parpadeo
TIEMPO_MIN_CIERRE = 1.0  # 1 segundo

# Definir el frame rate estimado (puede ajustarse según la cámara)
FRAME_RATE_ESTIMADO = 30  # 30 FPS

# Estado de los ojos para detección de cierre prolongado
estado_ojos = 'esperando_cierre'

# Variables para la detección de cabezadas hacia abajo
contador_cabezadas = 0
UMBRAL_PITCH_CABEZA = -15  # Umbral de pitch para detectar cabezada hacia abajo (ajustar según necesidad)
tiempo_ultima_cabezada = 0
COOLDOWN_CABEZA = 2.0  # Tiempo en segundos para evitar múltiples conteos por una sola cabezada

# Variables para la detección de cabezadas después de cerrar los ojos
cabezada_listo = False  # Indica si el sistema está listo para detectar una cabezada después de un cierre

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener el tiempo actual
    tiempo_actual = time.time()

    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    pitch, yaw, roll = None, None, None  # Inicializar variables de ángulos

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

            # Detección de parpadeo
            if EAR_promedio < EAR_UMBRAL:
                contador_frames_parpadeo += 1
            else:
                if contador_frames_parpadeo >= CONSEC_FRAMES_PARPADEO:
                    contador_parpadeos += 1
                contador_frames_parpadeo = 0  # Reiniciar contador

            # Detección de cierre prolongado de ojos
            if EAR_promedio < EAR_UMBRAL:
                if tiempo_inicio_cierre is None:
                    tiempo_inicio_cierre = tiempo_actual
            else:
                if tiempo_inicio_cierre is not None:
                    duracion_cierre = tiempo_actual - tiempo_inicio_cierre
                    if duracion_cierre >= TIEMPO_MIN_CIERRE:
                        total_tiempo_cerrado += duracion_cierre
                        cabezada_listo = True  # El sistema está listo para detectar una cabezada
                    tiempo_inicio_cierre = None  # Reiniciar

            # Estimar la pose de la cabeza
            pitch, yaw, roll = estimar_pose_cabeza(landmarks, frame.shape)

            # Detectar cabezada hacia abajo solo si el sistema está listo (después de un cierre de ojos)
            if cabezada_listo and pitch is not None:
                # Verificar la convención de los signos del pitch
                # Asumiendo que pitch negativo indica inclinación hacia abajo
                if pitch < UMBRAL_PITCH_CABEZA and (tiempo_actual - tiempo_ultima_cabezada) > COOLDOWN_CABEZA:
                    contador_cabezadas += 1
                    tiempo_ultima_cabezada = tiempo_actual
                    cabezada_listo = False  # Reiniciar para evitar múltiples conteos

    # Mostrar el conteo de parpadeos, tiempo cerrado y cabezadas
    cv2.putText(frame, f'Parpadeos: {contador_parpadeos}', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if pitch is not None:
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Tiempo Cerrado: {total_tiempo_cerrado:.2f} s', (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f'Cabezadas: {contador_cabezadas}', (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Opcional: Mostrar una línea o indicador de la cabeza
    if pitch is not None:
        # Dibuja una línea que indique la inclinación de la cabeza
        longitud_linea = 100
        punto_central = (w // 2, h // 2)
        # Ajuste en la dirección de la línea basado en la convención de pitch
        end_point = (
            int(punto_central[0] + longitud_linea * math.sin(math.radians(pitch))),
            int(punto_central[1] + longitud_linea * math.sin(math.radians(pitch)))
        )
        cv2.line(frame, punto_central, end_point, (255, 0, 0), 2)

    # Mostrar el frame
    cv2.imshow('Deteccion de Parpadeos, Tiempo Ojos Cerrados y Cabezadas', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Al finalizar, si los ojos estaban cerrados, sumar el tiempo restante
if tiempo_inicio_cierre is not None:
    duracion_cierre = time.time() - tiempo_inicio_cierre
    if duracion_cierre >= TIEMPO_MIN_CIERRE:
        total_tiempo_cerrado += duracion_cierre

# Mostrar el total de tiempo con ojos cerrados y cabezadas al finalizar
print(f'Total de tiempo con ojos cerrados: {total_tiempo_cerrado:.2f} segundos')
print(f'Total de cabezadas hacia abajo: {contador_cabezadas}')

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
