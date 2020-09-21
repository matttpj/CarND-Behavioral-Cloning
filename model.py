import csv
import cv2
import numpy as np


def get_rows_from_driving_log(dataPath, skipHeader=False):
    # Return multiple rows from the driving log at the base dataPath directory
    # If the file includes a header row then pass skipHeader=True

    rows = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for row in reader:
            rows.append(row)
    return rows


def load_image_control_values(dataPath, imagePath, control_value, images, control_values):
    # Execute the following steps:
    #  - Load the image at dataPath and imagePath
    #  - Convert the image from BGR to RGB
    #  - Add the image and control value to images[] and control_values[] lists
    #  - Flip the image vertically
    #  - Invert the sign of the control value
    #  - Add the flipped image and inverted control value to images and control values

    original_image = cv2.imread(dataPath + '/' + imagePath.strip())
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    images.append(image)
    control_values.append(control_value)
    # Flip the image and append new image to images[] list
    images.append(cv2.flip(image,1))
    # Reverse the control value and append to control_values[] list
    control_values.append(control_value*-1.0)

def load_images_controls(dataPath, skipHeader=False, correction=0.2):
    # Load the image paths and control values from the driving log in the base directory dataPath
    # If the driving log file includes a header row, pass skipHeader=True
    # Read each row and process image path and control value
    #  - center image plus control value (steering | throttle | brake | speed)
    #  - left | right images + or - correction on control value
    # Correction is the value to add/substract to the control value when using left/right side cameras
    # Sequentially build lists of images[] and control_values[]
    # Return two numpy arrays of images[], control_values[] as processing is more memory efficient

    rows = get_rows_from_driving_log(dataPath, skipHeader)
    images = []
    control_values = []

    #rows = rows[1:]

    for row in rows:
        control_value = float(row[3])
        # center
        load_image_control_values(dataPath, row[0], control_value, images, control_values)
        # left
        load_image_control_values(dataPath, row[1], control_value + correction, images, control_values)
        # right
        load_image_control_values(dataPath, row[2], control_value - correction, images, control_values)

    return (np.array(images), np.array(control_values))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def train_save(model, inputs, outputs, modelFile, epochs=5):
    # Train the model using loss='mse' and optimizer='adam' for epochs=5
    # Split initial input data 80% for model training and 20% for model testing
    # Shuffle the input images
    # The model is saved at modelFile

    model.compile(loss='mse', optimizer='adam')
    model.fit(inputs, outputs, validation_split=0.2, shuffle=True, epochs=5)
    model.save(modelFile)
    print("Model saved at " + modelFile)

def create_preprocessing_layers():
    # Create a Keras Sequential model with initial preprocessing layers
    # Normalise input data
    # Crop input images by ? px top + bottom and ? px left + right

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def leNet_model():
    # Create a LeNet model

    model = create_preprocessing_layers()
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nVidia_model():
    # Create an nVidia Autonomous Car model

    model = create_preprocessing_layers()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

print('Loading images and control values')
X_train, y_train = load_images_controls('data/data', skipHeader=True)
#model = leNet_model()
model = nVidia_model()

print('Training and saving model')
#train_save(model, X_train, y_train, 'models/model_leNet_3.h5')
train_save(model, X_train, y_train, 'models/model_nVidia_8.h5')
print('The End')
