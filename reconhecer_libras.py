import cv2
import mediapipe as mp
import joblib
import numpy as np

# Carrega o modelo
modelo = joblib.load("modelo_libras.pkl")

# Inicializa MediaPipe
mp_maos = mp.solutions.hands
maos = mp_maos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_desenho = mp.solutions.drawing_utils

# Inicia a câmera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(imagem_rgb)

    letra_prevista = ''

    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

            # Extrai os pontos (somente X e Y)
            pontos = []
            for lm in mao.landmark:
                pontos.extend([lm.x, lm.y])

            # Garante que tem os 21 pontos (42 valores)
            if len(pontos) == 42:
                entrada = np.array(pontos).reshape(1, -1)
                letra_prevista = modelo.predict(entrada)[0]

    # Exibe a letra reconhecida
    cv2.putText(frame, f"Letra: {letra_prevista}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Reconhecimento de Libras", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()