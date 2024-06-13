import cv2
import mediapipe as mp

# Inicialize os módulos de MediaPipe para Holistic (pose e mãos)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Configure o objeto de detecção Holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, 
                                smooth_landmarks=True, enable_segmentation=False, 
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture o vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Não foi possível capturar a imagem da câmera")
        break

    # Converta a imagem para RGB (MediaPipe usa RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inverter a imagem horizontalmente para um espelho de visualização
    image = cv2.flip(image, 1)

    # Para melhorar a performance, desative a gravação da imagem
    image.flags.writeable = False
    
    # Processar a imagem e encontrar a pose e mãos
    results = holistic.process(image)
    
    # Gravação da imagem ativada
    image.flags.writeable = True

    # Converta a imagem de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenhe as anotações da pose e mãos na imagem
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Mostrar a imagem
    cv2.imshow('Holistic Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
