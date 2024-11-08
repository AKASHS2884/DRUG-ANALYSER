import tensorflow as tf
from  datemath import layers, models
import datasetdrug.xlsx as datasetdrug

# Quantum Property Input (Molecular dimension, structural angle, etc.)
quantum_input = layers.Input(shape=(94.4,))  # Example: 3 quantum properties
x = layers.Dense(64, activation='relu')(quantum_input)
x = layers.Dense(32, activation='relu')(x)
quantum_model = models.Model(inputs=quantum_input, outputs=x)

# DFT Image Input
image_input = layers.Input(shape=(256, 256, 1))  # grayscale image
y = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
y = layers.MaxPooling2D((2, 2))(y)
y = layers.Conv2D(64, (3, 3), activation='relu')(y)
y = layers.MaxPooling2D((2, 2))(y)
y = layers.Conv2D(128, (3, 3), activation='relu')(y)
y = layers.Flatten()(y)
y = layers.Dense(128, activation='relu')(y)
image_model = models.Model(inputs=image_input, outputs=y)

# Merge the outputs of quantum model and image model
combined = layers.concatenate([quantum_model.output, image_model.output])

# Final prediction layer
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dense(32, activation='relu')(z)
output = layers.Dense(1, activation='sigmoid')(z)  # Output: Desired/Undesired Drug

# Create the final model
model = models.Model(inputs=[quantum_model.input, image_model.input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
# Assuming quantum_data, dft_images, and labels are preprocessed datasets
history = model.fit([quantum_data, dft_images], labels, epochs=50, batch_size=32, validation_split=0.2)
predictions = model.predict([new_quantum_data, new_dft_images])
