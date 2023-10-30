import tensorflow as tf
from tensorflow import keras

# Load a dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple neural network
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")
import tensorflow as tf
from tensorflow import keras

# Load the trained model (you can use the saved model from the previous code)
model = keras.models.load_model('path_to_your_saved_model')

# Load or prepare new data for prediction
# For example, if you have a new image for classification:
import numpy as np
from PIL import Image

# Load the image and preprocess it (similar to training data preprocessing)
image = Image.open('path_to_your_image.jpg')
image = np.array(image) / 255.0  # Normalize the image data

# Make predictions
predictions = model.predict(np.expand_dims(image, axis=0))

# Get the predicted class (assuming it's a classification task)
predicted_class = np.argmax(predictions)

# Print the predicted class
print(f"Predicted class: {predicted_class}")
import tensorflow as tf
from tensorflow import keras

# ... (previous code for training the model)

# Save the trained model
model.save('path_to_save_model')

# Now, let's assume you want to load the saved model and use it for predictions:

# Load the saved model
loaded_model = keras.models.load_model('path_to_save_model')

# Load or prepare new data for prediction
# For example, if you have new data for prediction:
import numpy as np

# Prepare the new data (similar to how you preprocessed the training data)
new_data = np.array(...)  # Replace ... with your new data

# Make predictions using the loaded model
predictions = loaded_model.predict(new_data)

# Handle the predictions as needed for your specific use case