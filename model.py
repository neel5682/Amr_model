import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv1D, BatchNormalization, Dropout, Sequential
import matplotlib.pyplot as plt
import numpy as np

# Define a custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize weights and bias for the attention mechanism
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Compute the attention scores
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        # Apply the attention scores to the input
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Define the model architecture
model = Sequential()
# Add the first Conv1D layer
model.add(Conv1D(256, kernel_size=8, input_shape=(128, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Add the second Conv1D layer
model.add(Conv1D(256, kernel_size=8, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Add the third Conv1D layer
model.add(Conv1D(80, kernel_size=8, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Add the custom Attention layer
model.add(Attention())
# Add a Dense layer with SELU activation and L2 regularization
model.add(Dense(256, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# Add the output Dense layer with softmax activation
model.add(Dense(11, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print the model summary
model.summary()

# Set batch size and number of epochs for training
batch_size = 1024  # Adjust batch size if needed
nb_epoch = 300  # Adjust number of epochs if needed

# Define the file path for saving the best model
filepath = '/kaggle/working/convmodrecnets_CNN2_0.5.wts.keras'
# Define callbacks for training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto')
]

# Perform training on the dataset
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks)

# Reload the best weights once training is finished
model.load_weights(filepath)

# Evaluate the model on the test set
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

