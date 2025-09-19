import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Carregar modelo de qualidade
modelo_qualidade = load_model("meu_modelo.h5")
labels_qualidade = ["boa", "ruim"]

# Carregar modelo pré-treinado para classificação geral
modelo_detector = load_model("modelo_detector_banana.h5")

# Abrir webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Etapa 1: Detectar banana ---
    img = cv2.resize(frame, (240, 240))  # mesmo input usado no treino
    x = np.expand_dims(img.astype("float32")/255.0, axis=0)

    pred_banana = modelo_detector.predict(x)[0][0]

    if pred_banana < 0.7:  # threshold simples
        # --- Etapa 2: Classificação boa/ruim ---
        banana_img = cv2.resize(frame, (240, 240))  # input do modelo de qualidade
        banana_img = np.expand_dims(banana_img.astype("float32")/255.0, axis=0)

        pred_qual = modelo_qualidade.predict(banana_img)[0]
        idx = np.argmax(pred_qual)
        label_qual = labels_qualidade[idx]
        prob_qual = pred_qual[idx]

        texto = f"Banana: {label_qual} ({prob_qual*100:.1f}%)"
        cor = (0,255,0) if label_qual == "boa" else (0,0,255)
    else:
        texto = "Nenhuma banana detectada"
        cor = (0,255,255)

    # Mostrar na tela
    cv2.putText(frame, texto, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    cv2.imshow("Detecção e Qualidade", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
