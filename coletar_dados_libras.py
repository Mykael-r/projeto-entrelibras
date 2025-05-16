import cv2
import mediapipe as mp
import csv
import os

mp_maos = mp.solutions.hands
maos = mp_maos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_desenho = mp.solutions.drawing_utils

arquivo_csv = "dados_libras.csv"
if not os.path.exists(arquivo_csv):
    with open(arquivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['letra']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)

camera = cv2.VideoCapture(0)
letra_atual = ''

print("Pressione a letra correspondente no teclado para salvar uma amostra. Pressione ESC para sair.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

            pontos = []
            for lm in mao.landmark:
                pontos.extend([lm.x, lm.y])  # Salva apenas X e Y

            if letra_atual and len(pontos) == 42:
                with open(arquivo_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([letra_atual] + pontos)
                print(f"Amostra da letra '{letra_atual}' salva com sucesso!")
                letra_atual = ''  # Reseta após salvar

    cv2.putText(frame, f"Letra atual: {letra_atual}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Coletor de Dados - Libras", frame)

    tecla = cv2.waitKey(1)
    if tecla == 27:  # ESC
        break
    elif 97 <= tecla <= 122:  # Letras minúsculas a-z
        letra_atual = chr(tecla).upper()

camera.release()
cv2.destroyAllWindows()
