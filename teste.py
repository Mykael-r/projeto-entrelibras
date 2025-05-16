import cv2
import mediapipe as mp
import time

mp_maos = mp.solutions.hands
maos = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_desenho = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado = maos.process(imagem_rgb)

    if resultado.multi_hand_landmarks and resultado.multi_handedness:
        y_base = 60  # Posição inicial do texto
        y_step = 40  # Espaçamento entre as linhas
        for idx, mao in enumerate(resultado.multi_hand_landmarks):

            tipo_mao = resultado.multi_handedness[idx].classification[0].label

            # Desenha os pontos e conexões da mão
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

            # Lista para armazenar as coordenadas dos pontos
            pontos = []
            for id, lm in enumerate(mao.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                pontos.append((cx, cy))

                # Mostra apenas o ponto 0 no terminal (centro da palma)
                if id == 0:
                    print(f"{tipo_mao} - Centro da palma: ({cx}, {cy})")

                # Desenha o número do ponto na tela
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Verifica se os 21 pontos foram capturados
            if len(pontos) == 21:
                dedos_levantados = 0

                # Polegar (comparação do eixo X)
                if tipo_mao == "Right":  # Para mão direita
                    if pontos[4][0] > pontos[3][0] and abs(pontos[4][1] - pontos[3][1]) > 20:
                        dedos_levantados += 1
                else:
                    if pontos[4][0] < pontos[3][0] and abs(pontos[4][1] - pontos[3][1]) > 20: # Para mão esquerda
                        dedos_levantados += 1

                # Outros dedos (comparação do eixo Y)
                for i in [8, 12, 16, 20]:  # Índices das pontas dos dedos
                    if pontos[i][1] < pontos[i - 2][1]:  # Verifica se está acima da articulação
                        dedos_levantados += 1

                # Exibe a contagem de dedos levantados
                texto = f"{tipo_mao}: {dedos_levantados} dedo(s) levantado(s)"
                cv2.putText(frame, texto, (10, y_base + y_step * idx * 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Identifica gestos simples
                if dedos_levantados == 5:
                    gesto = "Mão aberta"
                elif dedos_levantados == 0:
                    gesto = "Punho fechado"
                elif dedos_levantados == 1:
                    gesto = "Apenas 1 dedo"
                else:
                    gesto = f"{dedos_levantados} dedos"

                # Exibe o gesto reconhecido
                cv2.putText(frame, gesto, (10, y_base + y_step * (idx * 2 + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Detecção de Mãos - Libras IA", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.03)

camera.release()
cv2.destroyAllWindows()