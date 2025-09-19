import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# --- Carregar modelos TFLite ---
interpreter_detector = tflite.Interpreter(model_path="modelo_detector_banana.tflite")
interpreter_detector.allocate_tensors()

interpreter_qualidade = tflite.Interpreter(model_path="meu_modelo.tflite")
interpreter_qualidade.allocate_tensors()

labels_qualidade = ["boa", "ruim"]

# --- Função para inferência TFLite ---
def predict_tflite(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preparar input
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- Abrir webcam ---
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduzir tamanho para acelerar inferência
    img = cv2.resize(frame, (160, 160))

    # --- Etapa 1: Detectar banana ---
    pred_banana = predict_tflite(interpreter_detector, img)[0]

    if pred_banana < 0.7:
        # --- Etapa 2: Classificação boa/ruim ---
        banana_img = cv2.resize(frame, (160, 160))
        pred_qual = predict_tflite(interpreter_qualidade, banana_img)

        idx = np.argmax(pred_qual)
        label_qual = labels_qualidade[idx]
        prob_qual = pred_qual[idx]

        texto = f"Banana: {label_qual} ({prob_qual*100:.1f}%)"
        cor = (0, 255, 0) if label_qual == "boa" else (0, 0, 255)
    else:
        texto = "Nenhuma banana detectada"
        cor = (0, 255, 255)

    # --- Calcular FPS ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Mostrar texto da detecção
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    cv2.imshow("Detecção e Qualidade", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
