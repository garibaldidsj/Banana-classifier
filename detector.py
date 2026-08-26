import cv2
import os
import numpy as np
import tensorflow as tf

from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Modelo YOLO
YOLO_MODEL = "yolo11n.pt"

# Modelo MobileNetV2
QUALITY_MODEL = "models/quality_mobilenetv2.keras"

# Webcam
CAMERA_INDEX = 0

# Tamanho utilizado pela MobileNetV2
IMG_SIZE = (224, 224)

# Confiança mínima do YOLO
YOLO_CONFIDENCE = 0.50

# Confiança mínima para aceitar a classificação
QUALITY_CONFIDENCE = 0.50

# Classes da MobileNetV2
CLASS_NAMES = [
    "comestivel",
    "verde",
    "podre"
]

CLASS_COLORS = {
    "verde": (0, 255, 0),        # Verde
    "comestivel": (0, 255, 255), # Amarelo
    "podre": (0, 0, 255),        # Vermelho
    "desconhecido": (255, 255, 255) # Branco
}



# ============================================================
# CARREGAR MODELOS
# ============================================================

print("=" * 60)
print("BANANA CLASSIFIER")
print("=" * 60)

print("\nCarregando YOLOv8n...")

if not os.path.exists(YOLO_MODEL):
    raise FileNotFoundError(
        f"Modelo YOLO não encontrado: {YOLO_MODEL}"
    )

detector = YOLO(YOLO_MODEL)

print("YOLO carregado com sucesso.")
# ============================================================
# LOCALIZAR CLASSE BANANA
# ============================================================

banana_class_id = None

for class_id, class_name in detector.names.items():

    if class_name.lower() == "banana":
        banana_class_id = class_id
        break

if banana_class_id is None:
    raise RuntimeError(
        "A classe 'banana' não foi encontrada no modelo YOLO."
    )

print(
    f"Classe 'banana' encontrada no YOLO: "
    f"{banana_class_id}"
)

print("\nCarregando MobileNetV2...")

if not os.path.exists(QUALITY_MODEL):
    raise FileNotFoundError(
        f"Modelo MobileNetV2 não encontrado: {QUALITY_MODEL}"
    )

quality_model = tf.keras.models.load_model(
    QUALITY_MODEL
)

print("MobileNetV2 carregada com sucesso.")


# ============================================================
# CLASSIFICAÇÃO DA QUALIDADE
# ============================================================

def classify_quality(crop):
    """
    Classifica a qualidade da banana utilizando a MobileNetV2.

    A decisão prioriza as classes verde e podre para evitar
    que frutas que apresentam características dessas classes
    sejam classificadas como comestíveis.
    """

    if crop is None or crop.size == 0:
        return "desconhecido", 0.0

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    # Redimensionar
    image = cv2.resize(
        image,
        IMG_SIZE
    )

    # Converter para float32
    image = image.astype(np.float32)

    # Normalização
    image = image / 255.0

    # Adicionar dimensão do batch
    image = np.expand_dims(
        image,
        axis=0
    )

    # Predição
    prediction = quality_model.predict(
        image,
        verbose=0
    )[0]

    # Probabilidades
    comestivel_prob = float(prediction[0])
    verde_prob = float(prediction[1])
    podre_prob = float(prediction[2])

    # ========================================================
    # REGRAS DE DECISÃO
    # ========================================================

    # Se PODRE tiver uma probabilidade significativa,
    # priorizamos essa classificação.
    if podre_prob >= 0.25:

        label = "podre"
        confidence = podre_prob

    # Se VERDE tiver uma probabilidade significativa,
    # priorizamos essa classificação.
    elif verde_prob >= 0.25:

        label = "verde"
        confidence = verde_prob

    # Caso contrário, utiliza a maior probabilidade.
    else:

        class_id = int(
            np.argmax(prediction)
        )

        label = CLASS_NAMES[class_id]
        confidence = float(
            prediction[class_id]
        )

    # ========================================================
    # CLASSIFICAÇÃO INCERTA
    # ========================================================

    if confidence < QUALITY_CONFIDENCE:

        label = "desconhecido"

    return label, confidence


# ============================================================
# DETECÇÃO
# ============================================================

def detect_banana(frame):
    """
    Detecta somente bananas utilizando o YOLO11n.
    Todos os demais objetos são ignorados.
    """

    results = detector.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        classes=[banana_class_id],
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].cpu().numpy()
            )

            # Segurança: ignora qualquer objeto
            # que não seja banana
            if class_id != banana_class_id:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence
            })

    return detections
    """
    Detecta bananas utilizando YOLOv8n.

    Retorna:
        lista de detecções contendo:
        x1, y1, x2, y2, confiança e classe
    """

    results = detector.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # Coordenadas
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id
            })

    return detections


# ============================================================
# PROCESSAMENTO DO FRAME
# ============================================================



def process_frame(frame):

    # ========================================================
    # DETECTAR SOMENTE BANANAS
    # ========================================================

    detections = detect_banana(frame)

    # ========================================================
    # PROCESSAR CADA BANANA DETECTADA
    # ========================================================

    for detection in detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        detection_conf = detection["confidence"]

        # ====================================================
        # DIMENSÕES DA IMAGEM
        # ====================================================

        h, w = frame.shape[:2]

        # Garantir que as coordenadas estejam dentro da imagem
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))

        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # ====================================================
        # VERIFICAR BOUNDING BOX
        # ====================================================

        if x2 <= x1 or y2 <= y1:
            continue

        # ====================================================
        # RECORTAR SOMENTE A BANANA
        # ====================================================

        crop = frame[
            y1:y2,
            x1:x2
        ]

        # ====================================================
        # CLASSIFICAR QUALIDADE
        # ====================================================

        quality, quality_conf = classify_quality(
            crop
        )

        # ====================================================
        # DEFINIR TEXTO
        # ====================================================

        if quality == "comestivel":

            display_label = (
                f"COMESTIVEL "
                f"{quality_conf * 100:.1f}%"
            )

        elif quality == "verde":

            display_label = (
                f"VERDE "
                f"{quality_conf * 100:.1f}%"
            )

        elif quality == "podre":

            display_label = (
                f"PODRE "
                f"{quality_conf * 100:.1f}%"
            )

        else:

            display_label = (
                f"DESCONHECIDO "
                f"{quality_conf * 100:.1f}%"
            )

        # ====================================================
        # DEFINIR COR
        #
        # OpenCV utiliza BGR:
        #
        # Verde      = (0, 255, 0)
        # Amarelo    = (0, 255, 255)
        # Vermelho   = (0, 0, 255)
        # Branco     = (255, 255, 255)
        # ====================================================

        if quality == "verde":

            color = (0, 255, 0)

        elif quality == "comestivel":

            color = (0, 255, 255)

        elif quality == "podre":

            color = (0, 0, 255)

        else:

            color = (255, 255, 255)

        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # ====================================================
        # TEXTO DA CLASSIFICAÇÃO
        # ====================================================

        cv2.putText(
            frame,
            display_label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # ====================================================
        # CONFIANÇA DO YOLO
        # ====================================================

        detector_text = (
            f"BANANA: "
            f"{detection_conf * 100:.1f}%"
        )

        cv2.putText(
            frame,
            detector_text,
            (x1, min(h - 10, y2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # ========================================================
    # RETORNAR FRAME
    # ========================================================

    return frame


# ============================================================
# WEBCAM
# ============================================================

print("\nInicializando webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    raise RuntimeError(
        "Não foi possível abrir a webcam."
    )

print("Webcam iniciada.")
print("\nPressione 'q' para sair.\n")


# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Erro ao capturar frame."
        )

        break

    # Processar frame
    frame = process_frame(
        frame
    )

    # Mostrar
    cv2.imshow(
        "Banana Classifier",
        frame
    )

    # Tecla Q
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# FINALIZAÇÃO
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nSistema finalizado.")