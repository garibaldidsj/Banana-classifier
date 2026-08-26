import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURAÇÕES
# ============================================================

TEST_DIR = "dataset/quality/test"
MODEL_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

CLASS_NAMES = [
    "comestivel",
    "verde",
    "podre"
]


# ============================================================
# LOCALIZAR MODELO
# ============================================================

def find_model():
    extensions = ["*.keras", "*.h5"]

    models = []

    for extension in extensions:
        models.extend(glob.glob(
            os.path.join(MODEL_DIR, "**", extension),
            recursive=True
        ))

    if not models:
        print("\nERRO: nenhum modelo .keras ou .h5 encontrado.")
        print(f"Verifique a pasta: {MODEL_DIR}")
        exit(1)

    if len(models) > 1:
        print("\nModelos encontrados:")

        for i, model in enumerate(models):
            print(f"[{i}] {model}")

        print("\nUtilizando o primeiro modelo encontrado.")

    return models[0]


# ============================================================
# VERIFICAR DATASET
# ============================================================

def check_dataset():

    if not os.path.exists(TEST_DIR):
        print(f"\nERRO: dataset não encontrado:")
        print(TEST_DIR)
        exit(1)

    print("\nDataset encontrado:")
    print(TEST_DIR)

    total = 0

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(TEST_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"ERRO: classe não encontrada: {class_dir}")
            exit(1)

        images = []

        for extension in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(
                glob.glob(
                    os.path.join(class_dir, extension)
                )
            )

        print(f"  {class_name}: {len(images)} imagens")

        total += len(images)

    print(f"\nTotal de imagens: {total}")

    return total


# ============================================================
# CARREGAR DATASET
# ============================================================

def load_dataset():

    print("\nCarregando dataset...")

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Normalização
    normalization = tf.keras.layers.Rescaling(
        1.0 / 255
    )

    dataset = dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate_model(model, dataset):

    print("\n" + "=" * 60)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 60)

    # Avaliação através do Keras
    results = model.evaluate(
        dataset,
        verbose=1
    )

    print("\nResultados do modelo:")

    for name, value in zip(
        model.metrics_names,
        results
    ):
        print(f"{name}: {value:.4f}")

    # ========================================================
    # PREDIÇÕES
    # ========================================================

    print("\nGerando predições...")

    y_true = []
    y_pred = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            np.argmax(
                labels.numpy(),
                axis=1
            )
        )

        y_pred.extend(
            np.argmax(
                predictions,
                axis=1
            )
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nMatriz de confusão:")
    print(cm)

    # ========================================================
    # PLOT
    # ========================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Matriz de Confusão - MobileNetV2"
    )

    plt.tight_layout()

    output_file = "confusion_matrix_quality.png"

    plt.savefig(
        output_file,
        dpi=300
    )

    print(
        f"\nMatriz de confusão salva em: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTE DO CLASSIFICADOR DE QUALIDADE")
    print("=" * 60)

    # Localizar modelo
    model_path = find_model()

    print(f"\nModelo utilizado:")
    print(model_path)

    # Verificar dataset
    check_dataset()

    # Carregar modelo
    print("\nCarregando modelo...")

    model = tf.keras.models.load_model(
        model_path
    )

    print("Modelo carregado com sucesso.")

    # Carregar dataset
    test_dataset = load_dataset()

    # Avaliar
    evaluate_model(
        model,
        test_dataset
    )

    print("\n" + "=" * 60)
    print("TESTE FINALIZADO")
    print("=" * 60)