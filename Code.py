!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
from google.colab import drive
drive.mount('/content/drive')
!kaggle datasets download -d brsdincer/wildfire-detection-image

import zipfile
zip_ref = zipfile.ZipFile('/content/wildfire-detection-image.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

"""
# **Forest Fire Detection Using Convolutional Neural Network**
link to dataset: https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

"""Making separate datasets for training and testing"""

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    "/content/forest_fire/Training and Validation/",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

test_dataset = test.flow_from_directory(
    "/content/forest_fire/Testing/",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

test_dataset.class_indices

"""Model Building"""

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

"""Compiling the model"""

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""Fitting the model"""

r = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

"""Predicting on Test Dataset"""

model.save("/content/drive/MyDrive/FinalModelDirectory")
predictions = model.predict(test_dataset)
predictions = np.round(predictions)
print(len(predictions))

"""Plotting loss per iteration"""

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.plot(r.history['accuracy'], label='accuracy')
plt.legend()

"""Plotting accuracy per iteration"""

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

"""Making a function to see any image from dataset with predicted label"""

test_dataset.classes > 0.5
predicted_labels = predictions.flatten() > 0.5
predicted_labels

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your true labels stored in a NumPy array named 'true_labels'
true_labels = test_dataset.classes  # Replace with your actual true labels

# Convert predictions to class labels (0 or 1)
predicted_labels = predictions.flatten() > 0.5

# Calculate the confusion matrix and classification report
confusion_mat = confusion_matrix(true_labels, predicted_labels)
classification_rep = classification_report(true_labels, predicted_labels)

# Print the classification report
print("Classification Report:\n", classification_rep)

# Plot the confusion matrix as a heatmap with a red theme
sns.heatmap(confusion_mat, annot=True, cmap="Reds")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

def predictImage(filename):
    img1 = image.load_img(filename, target_size=(150,150))
    plt.imshow(img1)
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    print(val)
    if val == 1:
        plt.xlabel("No Fire", fontsize=30)
    elif val == 0:
        plt.xlabel("Fire", fontsize=30)

predictImage("/content/forest_fire/Testing/fire/abc182.jpg")
predictImage('/content/forest_fire/Testing/fire/abc190.jpg')
predictImage('/content/forest_fire/Testing/nofire/abc346.jpg')
predictImage('/content/forest_fire/Testing/nofire/abc361.jpg')
predictImage('/content/forest_fire/Training and Validation/fire/abc011.jpg')
predictImage('/content/forest_fire/Testing/fire/abc172.jpg')
predictImage('/content/forest_fire/Testing/nofire/abc341.jpg')

# CONVERTING TO THE TFTLITE MODEL !!!

import glob
import os
import pandas as pd
import numpy as np

model = tf.keras.models.load_model("/content/drive/MyDrive/FinalModelDirectory/Model_to_Be_shown.h5")  # Replace with your model's path
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = "/content/drive/MyDrive/FinalModelDirectory/Final_Quantized_Model_ToBeShown.tflite"
open(tflite_path, "wb").write(tflite_model)

# HERE QUANTIZATION IS APPLIED IN ORDER TO FURTHER COMPRESS THE MODEL !!

converter = tf.lite.TFLiteConverter.from_saved_model("/content/drive/MyDrive/FinalModelDirectory")
model_no_quant_tflite = converter.convert()

# Save the model to disk
open("float.tflite", "wb").write(model_no_quant_tflite)

def representative_dataset():
    for filename in glob.glob("/content/forest_fire/Testing" + "/*/*.jpg"):
        img = keras.preprocessing.image.load_img(filename, target_size=(150, 150))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        for images, labels in train_ds.take(1):
            yield([img_array])

# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
# open("quantized.tflite", "wb").write(model_tflite)
tflite_path = "/content/drive/MyDrive/FinalModelDirectory/CompressedModel_ToBeShown.tflite"
open(tflite_path, "wb").write(model_tflite)

"""## Comparing Sizes Between Models """

def get_dir_size(dir):
    size = 0
    for f in os.scandir(dir):
        if f.is_file():
            size += f.stat().st_size
        elif f.is_dir():
            size += get_dir_size(f.path)
    return size

# Calculate size
size_tf = get_dir_size("/content/drive/MyDrive/FinalModelDirectory")
size_no_quant_tflite = os.path.getsize("float.tflite")
size_tflite = os.path.getsize("/content/drive/MyDrive/FinalModelDirectory/CompressedModel_ToBeShown.tflite")

# Compare size
pd.DataFrame.from_records(
    [
        ["TensorFlow", f"{size_tf} bytes", ""],
        ["TensorFlow Lite", f"{size_no_quant_tflite} bytes ", f"(reduced by {size_tf - size_no_quant_tflite} bytes)"],
        ["TensorFlow Lite Quantized", f"{size_tflite} bytes", f"(reduced by {size_no_quant_tflite - size_tflite} bytes)"]
    ],
    columns=["Model", "Size", ""],
    index="Model"
)

def predict_tflite(tflite_model, filename):
    img = keras.preprocessing.image.load_img(filename, target_size=(150, 150))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # If required, quantize the input layer (from float to integer)
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        img_array = np.multiply(img_array, 1.0 / input_scale) + input_zero_point
        img_array = img_array.astype(input_details["dtype"])

    # Invoke the interpreter
    interpreter.set_tensor(input_details["index"], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details["index"])[0]

    # If required, dequantized the output layer (from integer to float)
    output_scale, output_zero_point = output_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        pred = pred.astype(np.float32)
        pred = np.multiply((pred - output_zero_point), output_scale)

    predicted_label_index = np.argmax(pred)
    predicted_score = pred[predicted_label_index]
    return predicted_score

print(predict_tflite(tflite_model, "/content/forest_fire/Testing/nofire/abc341.jpg"))

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('your_model.h5')

# Prepare input data for prediction
# Replace this with your own input data
input_data = np.array([[...]])

# Make predictions
predictions = model.predict(input_data)

# Output the predictions
print(predictions)
