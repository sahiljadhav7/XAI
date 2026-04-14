import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

# # Load the trained model
# model = load_model("model.h5")


def get_model_image_size(model, fallback=(256, 256)):
    shape = getattr(model, "input_shape", None)
    if not shape or len(shape) < 4:
        return fallback

    height = shape[1] or fallback[0]
    width = shape[2] or fallback[1]
    return int(height), int(width)


def resolve_last_conv_layer_name(model, preferred_layer_name=None):
    if preferred_layer_name:
        try:
            model.get_layer(preferred_layer_name)
            return preferred_layer_name
        except ValueError:
            pass

    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            return layer.name

    raise ValueError("No convolutional layer found for Grad-CAM generation.")

# Function to preprocess input image
def preprocess_image(image_path, img_size=(256, 256)):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, img_size)
    model_input = preprocess_input(image_rgb.astype(np.float32))
    return np.expand_dims(model_input, axis=0), image_rgb  # Return batch and original image

# Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    last_conv_layer_name = resolve_last_conv_layer_name(model, last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.math.reduce_max(heatmap)
    if tf.equal(max_value, 0):
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / max_value
    return heatmap.numpy()

# Function to overlay Grad-CAM heatmap on original image
def overlay_gradcam(image, heatmap, alpha=0.4, colormap=cm.jet):
    # Convert image to uint8 (if it's not already)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = np.uint8(heatmap_colored * 255)

    # Ensure both are uint8
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed_img

# Function to make a prediction and generate Grad-CAM output
def predict_and_visualize(image_path, model, last_conv_layer_name="mixed10"):
    img_array, original_image = preprocess_image(image_path, get_model_image_size(model))
    prediction = model.predict(img_array, verbose=0)

    if prediction.shape[-1] == 1:
        positive_probability = float(prediction[0][0])
        predicted_class = 1 if positive_probability >= 0.5 else 0
        predicted_probability = positive_probability if predicted_class == 1 else 1.0 - positive_probability
        gradcam_index = 0
    else:
        predicted_class = int(np.argmax(prediction[0]))
        predicted_probability = float(prediction[0][predicted_class])
        gradcam_index = predicted_class

    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, gradcam_index)
    gradcam_image = overlay_gradcam(original_image, heatmap)
    
    return predicted_class, predicted_probability, gradcam_image

# Function to get Grad-CAM heatmap without overlaying on original image
def generate_gradcam_only(image_path, model, last_conv_layer_name="mixed10"):
    img_array, _ = preprocess_image(image_path, get_model_image_size(model))
    prediction = model.predict(img_array, verbose=0)

    if prediction.shape[-1] == 1:
        predicted_class = 1 if float(prediction[0][0]) >= 0.5 else 0
        gradcam_index = 0
    else:
        predicted_class = int(np.argmax(prediction[0]))
        gradcam_index = predicted_class

    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, gradcam_index)
    return predicted_class, heatmap

# Function to check if the given image is a lung X-ray
def is_lung_xray(image_path, model):
    img = image.load_img(image_path, target_size=get_model_image_size(model))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize input

    prediction = model.predict(img_array, verbose=0)  # Get probability
    lung_prob = prediction[0][0]  # Assuming output is [lung_prob]

    return lung_prob > 0.5  # Return True if it's a lung X-ray, False otherwise
