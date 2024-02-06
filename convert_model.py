import tensorflow as tf

tf_model = tf.keras.models.load_model("model_files/sample.h5")

####### FP 16 ########
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = "model_files/sample.tflite"

with open(tflite_model_quantized_file, "wb") as f:
    f.write(tflite_model_quantized)
