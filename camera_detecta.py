import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Carregar modelo treinado ===
model = load_model("meu_modelo.h5")  # coloque o caminho do seu modelo

# Labels (ordem deve ser a mesma usada no treino)
labels = ["boa", "ruim"]  # ajuste conforme usou no treinamento

# === Abrir webcam ===
cap = cv2.VideoCapture(0)  # 0 = webcam padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Pré-processamento ===
    img = cv2.resize(frame, (240, 240))  # mesmo tamanho do treino
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # [1, 240, 240, 3]

    # === Predição ===
    pred = model.predict(img)
    class_index = np.argmax(pred[0])  # índice da classe
    prob = np.max(pred[0])            # confiança
    label = labels[class_index]

    # === Mostrar resultado na tela ===
    texto = f"{label} ({prob*100:.1f}%)"
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Classificação de Frutas", frame)

    # === Sair com tecla 'q' ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
