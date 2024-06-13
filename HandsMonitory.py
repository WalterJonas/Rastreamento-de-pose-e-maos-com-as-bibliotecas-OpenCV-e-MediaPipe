import cv2
import mediapipe as mp

# Inicialize os módulos de MediaPipe para mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure o objeto de detecção de mãos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

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
    
    # Processar a imagem e encontrar mãos
    results = hands.process(image)
    
    # Gravação da imagem ativada
    image.flags.writeable = True

    # Converta a imagem de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenhe as anotações da mão na imagem
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar a imagem
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
