from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims
from model import train, get_model
import os

def read_image(filename, shape):
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image

train(train_dataset_path='/dataset/train', test_dataset_path='/dataset/test', learning_rate=0.0005, epochs=10)

model = get_model()
model.load_weights('model.h5')

x = model.predict(read_image('validation/damage_screen.jpg', (150, 150)))
print('damaged' if x[0][0] < 0.5 else 'not damaged', 'accuracy: {0}'.format(x[0][0]))
