import h5py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet', include_top=True, classes=1000)
model.save('mobilenet_v3.h5')
# Load the model
model = tf.keras.models.load_model('mobilenet_v3.h5')

# Print model summary to see its architecture and parameters
print(model.summary())

# # Accessing the weights of each layer
# for layer in model.layers:
#     print("Layer Name:", layer.name)
#     weights = layer.get_weights() # Get the weights of the layer
#     if len(weights) > 0:
#         for i, w in enumerate(weights):
#             print(f"Weight {i} shape:", w.shape)
#     else:
#         print("No weights in this layer")
