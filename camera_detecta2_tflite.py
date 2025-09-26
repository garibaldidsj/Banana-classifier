import cv2
import numpy as np
import tensorflow as tf

# Carregar modelo de qualidade (TFLite)
interpreter_qualidade = tf.lite.Interpreter(model_path="meu_modelo.tflite")
interpreter_qualidade.allocate_tensors()

# Carregar modelo detector (TFLite)
interpreter_detector = tf.lite.Interpreter(model_path="modelo_detector_banana.tflite")
interpreter_detector.allocate_tensors()

# Labels
labels_qualidade = ["boa", "ruim"]

# Função para rodar inferência em um modelo TFLite
def predict_tflite(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Garantir formato correto
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Abrir câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Etapa 1: Detectar banana ---
    img = cv2.resize(frame, (240, 240))
    pred_banana = predict_tflite(interpreter_detector, img)[0]

    if pred_banana < 0.7:  # threshold simples
        # --- Etapa 2: Classificação boa/ruim ---
        banana_img = cv2.resize(frame, (240, 240))
        pred_qual = predict_tflite(interpreter_qualidade, banana_img)

        idx = np.argmax(pred_qual)
        label_qual = labels_qualidade[idx]
        prob_qual = pred_qual[idx]

        texto = f"Banana: {label_qual} ({prob_qual*100:.1f}%)"
        cor = (0, 255, 0) if label_qual == "boa" else (0, 0, 255)
    else:
        texto = "Nenhuma banana detectada"
        cor = (0, 255, 255)

    # Mostrar na tela
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    cv2.imshow("Detecção e Qualidade", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
