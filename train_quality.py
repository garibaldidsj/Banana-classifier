from pathlib import Path


import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

import tensorflow as tf
os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

tf.config.optimizer.set_jit(False)

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 64

INITIAL_EPOCHS = 10

FINETUNING_EPOCHS = 10

DATASET_DIR = Path("dataset/quality")

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASSES = [
    "comestivel",
    "verde",
    "podre"
]


# ============================================================
# GPU
# ============================================================

print("=" * 60)
print("CONFIGURAÇÃO DE HARDWARE")
print("=" * 60)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print(f"\nGPU(s) encontrada(s): {len(gpus)}")

    for gpu in gpus:

        print(" ", gpu)

    # Evita que o TensorFlow reserve toda a memória
    # da GPU de uma vez.

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        print("\nMemory growth: habilitado")

    except RuntimeError as error:

        print(
            "\nNão foi possível configurar "
            "memory growth:"
        )

        print(error)

else:

    print("\nATENÇÃO!")
    print("Nenhuma GPU foi encontrada.")
    print("O treinamento será executado na CPU.")


print("=" * 60)


# ============================================================
# VERIFICAR DATASET
# ============================================================

print("\n")
print("=" * 60)
print("VERIFICANDO DATASET")
print("=" * 60)


def count_images(directory):

    if not directory.exists():
        return 0

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    return sum(

        1

        for p in directory.rglob("*")

        if p.is_file()
        and p.suffix.lower() in extensions
    )


for split in [
    "train",
    "val",
    "test"
]:

    print(f"\n{split.upper()}")

    total = 0

    for class_name in CLASSES:

        directory = (
            DATASET_DIR
            / split
            / class_name
        )

        count = count_images(
            directory
        )

        total += count

        print(
            f"  {class_name:12}: "
            f"{count} imagens"
        )

    print(
        f"  TOTAL: {total}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    horizontal_flip=True,

    brightness_range=(
        0.8,
        1.2
    )
)


val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================================
# DATASETS
# ============================================================

print("\n")
print("=" * 60)
print("CARREGANDO IMAGENS")
print("=" * 60)


train_dataset = train_datagen.flow_from_directory(

    DATASET_DIR / "train",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=True,

    seed=42
)


val_dataset = val_datagen.flow_from_directory(

    DATASET_DIR / "val",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=False
)


test_dataset = val_datagen.flow_from_directory(

    DATASET_DIR / "test",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=False
)


print("\nClasses:")

print(
    train_dataset.class_indices
)


# ============================================================
# MOBILE NET V2
# ============================================================

print("\n")
print("=" * 60)
print("CRIANDO MOBILENETV2")
print("=" * 60)


base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )
)


# Congela inicialmente

base_model.trainable = False


x = base_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

output = layers.Dense(

    3,

    activation="softmax"
)(x)


model = Model(

    inputs=base_model.input,

    outputs=output
)


# ============================================================
# COMPILAÇÃO
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-3
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-6
    ),

    ModelCheckpoint(

        MODEL_DIR
        / "quality_mobilenetv2.keras",

        monitor="val_accuracy",

        save_best_only=True
    )
]


# ============================================================
# TREINAMENTO INICIAL
# ============================================================

print("\n")
print("=" * 60)
print("TREINAMENTO INICIAL")
print("=" * 60)


model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n")
print("=" * 60)
print("FINE-TUNING")
print("=" * 60)


base_model.trainable = True


# Libera somente as últimas 40 camadas

for layer in base_model.layers[:-40]:

    layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINETUNING_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# AVALIAÇÃO
# ============================================================

print("\n")
print("=" * 60)
print("AVALIAÇÃO")
print("=" * 60)


results = model.evaluate(

    test_dataset,

    verbose=1
)


for name, value in zip(

    model.metrics_names,

    results

):

    print(
        f"{name}: {value:.4f}"
    )


print("\nModelo salvo em:")

print(
    MODEL_DIR
    / "quality_mobilenetv2.keras"
)