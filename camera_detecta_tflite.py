import serial
import time
import cv2
import numpy as np
import tensorflow.lite as tflite

# --- Conexão serial ---
arduino = serial.Serial('COM6', 9600)  # troque COM3 pela sua porta real
time.sleep(2)  # espera inicialização

# --- Carregar modelos ---
interpreter_detector = tflite.Interpreter(model_path="modelo_detector_banana.tflite")
interpreter_detector.allocate_tensors()

interpreter_qualidade = tflite.Interpreter(model_path="meu_modelo.tflite")
interpreter_qualidade.allocate_tensors()

labels_qualidade = ["boa", "ruim"]

def predict_tflite(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

cap = None
janela_aberta = False

boa_count = 0
ruim_count = 0

while True:
    # Sempre verificar mensagens do Arduino
    if arduino.in_waiting > 0:
        msg = arduino.readline().decode().strip()
        print(f"Mensagem do Arduino: {msg}")

        if msg == "detected" and not janela_aberta:
            print("Objeto detectado → iniciando visão computacional")
            cap = cv2.VideoCapture(0)
            janela_aberta = True
            boa_count = 0
            ruim_count = 0

        elif msg == "timeout" and janela_aberta:
            print("Timeout recebido → fechando janela")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            janela_aberta = False

            # Determinar resultado final
            resultado = "indefinido"
            if boa_count > ruim_count:
                resultado = "boa"
            elif ruim_count > boa_count:
                resultado = "ruim"

            print(f"Resultado final: {resultado} (boa={boa_count}, ruim={ruim_count})")

            # Enviar resultado ao Arduino
            arduino.write((resultado + "\n").encode())

    # Se a janela estiver aberta, processa frames
    if janela_aberta:
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.resize(frame, (240, 240))
        pred_banana = predict_tflite(interpreter_detector, img)[0]

        if pred_banana < 0.7:
            banana_img = cv2.resize(frame, (240, 240))
            pred_qual = predict_tflite(interpreter_qualidade, banana_img)

            idx = np.argmax(pred_qual)
            label_qual = labels_qualidade[idx]
            prob_qual = pred_qual[idx]
            texto = f"Banana: {label_qual} ({prob_qual*100:.1f}%)"
            cor = (0, 255, 0) if label_qual == "boa" else (0, 0, 255)

            # Contabilizar
            if label_qual == "boa":
                boa_count += 1
            elif label_qual == "ruim":
                ruim_count += 1
        else:
            texto = "Nenhuma banana detectada"
            cor = (0, 255, 255)

        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        cv2.imshow("Detecção e Qualidade", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Janela fechada manualmente")
            janela_aberta = False
            cap.release()
            cv2.destroyAllWindows()
