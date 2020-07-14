import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def show_plots(history, epochs):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def get_model(shape=(200, 200), learning_rate=0.001):
    model = Sequential([
        # convolutional layer
        Conv2D(32, (3, 3), input_shape=(shape[0], shape[1], 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # convert input to 1-D vector
        Flatten(),
        # dense hidden layer
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model
    

def train(train_dataset_path=None, test_dataset_path=None, batch_size=16, learning_rate=0.001, epochs=3, shape=(200, 200)):
    # chekc dataset paths
    if train_dataset_path == None or test_dataset_path == None:
        raise ValueError()

    # count train & test images
    train_len = sum([len(files) for _, _, files in os.walk(train_dataset_path)])
    test_len = sum([len(files) for _, _, files in os.walk(test_dataset_path)])
    
    # generate train images
    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dataset_path,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            target_size=shape,
                                                            class_mode='binary')
    
    # generate test images
    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(directory=test_dataset_path,
                                                            shuffle=True,
                                                            batch_size=batch_size,
                                                            target_size=shape,
                                                            class_mode='binary')
    
    # define model layers
    model = get_model(shape, learning_rate)
    
    # Configure gradient descent method & loss method 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # start learn
    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_len,
        epochs=epochs,
        validation_data=test_data_gen,
        validation_steps=test_len
    )

    # show learning accuracy & loss plots
    show_plots(history, epochs)

    # save model weights
    model.save_weights('model.h5')
