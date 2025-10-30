import tensorflow as tf
from tensorflow import keras

# Carregar o modelo que você já treinou
modelo_detector = keras.models.load_model("modelo_detector_banana.h5")

# Converter para TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo_detector)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # ops TFLite padrão
    tf.lite.OpsSet.SELECT_TF_OPS          # ops adicionais do TF
]
tflite_model = converter.convert()

# Salvar
with open("modelo_detector_banana.tflite", "wb") as f:
    f.write(tflite_model)
