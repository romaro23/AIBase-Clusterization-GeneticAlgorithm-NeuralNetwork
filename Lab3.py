import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
split_index = int(0.8 * len(x_train))
x_val, y_val = x_train[split_index:], y_train[split_index:]
x_train, y_train = x_train[:split_index], y_train[:split_index]

def train_val_plot(history_):
    loss     = history_.history['loss']
    val_loss = history_.history['val_loss']
    epochs   = range(len(loss))
    plt.figure()
    plt.plot  ( epochs,loss )
    plt.plot  ( epochs,val_loss )
    plt.title ('Training and validation loss')
    plt.show()
def acc_plot(history_):
    acc = history_.history['accuracy']
    val_acc  = history_.history[ 'val_accuracy' ]
    epochs   = range(len(acc))
    plt.figure()
    plt.plot  ( epochs,     acc )
    plt.plot  ( epochs, val_acc )
    plt.title ('Training and validation accuracy')
    plt.show()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=24, validation_data=(x_val, y_val))
train_val_plot(history)
acc_plot(history)
model.evaluate(x_test, y_test, verbose=2)
for i in range(5):
    plt.imshow(x_test[i], cmap='binary')
    plt.show()
    prediction = model.predict(x_test[i:i+1])
    print(f"Predicted probabilities for image {i}:")
    print(prediction)
    print(f"Predicted class: {tf.argmax(prediction[0])}")