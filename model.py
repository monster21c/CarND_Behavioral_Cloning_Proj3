import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math

input_shape = (160, 320, 3)
batch_size = 32
samples = []

with open('./data/driving_log.csv') as csvfile:  # for track 1
    # with open('./data_2/driving_log.csv') as csvfile:  # for track 2
    reader = csv.reader(csvfile)
    # ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    header = next(reader)  # <<- skip the headers
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


width, height = 320, 160


def img_choice(batch_sample):
    correction = 0.25
    img_choice = np.random.randint(3)
    steering_angle_center = float(batch_sample[3])

    if img_choice == 0:
        source_path = batch_sample[1]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename  # for track 2
        image = mpimg.imread(current_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
        steering_angle = steering_angle_center + correction
    elif img_choice == 1:
        source_path = batch_sample[2]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename  # for track 2
        image = mpimg.imread(current_path)
        steering_angle = steering_angle_center - correction
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
    elif img_choice == 2:
        source_path = batch_sample[0]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename  # for track 2
        image = mpimg.imread(current_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
        steering_angle = steering_angle_center
    return image, steering_angle


def rgb2yuv(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def resize_img(input):
    from keras.backend import tf as ktf
    # return ktf.image.resize_images(input, (128, 128))
    return ktf.image.resize_images(input, (64, 160))


def crop_img(image):
    """ crop unnecessary parts """
    cropped_img = image[65:135, 0:320]
    # reducing the size of image by 50%
    resized_img = cv2.resize(cropped_img, (width, height), cv2.INTER_AREA)
    # image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return image


def random_flip(image, steering_angle):
    flip_prob = np.random.rand()
    if flip_prob > 0.5:
        image = cv2.flip(image, 1)  # flip the images horizontally
        # image = np.fliplr(image)
        steering_angle = steering_angle * -1.0  # reverse the stearing angle
    return image, steering_angle


def random_shadow(image):
    right_x = width * np.random.uniform()
    top_y = 0
    bot_y = height
    left_x = width * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = np.zeros_like(image_hls[:, :, 1])

    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask = np.zeros_like(image[:, :, 1])
    shadow_mask[(Y_m - top_y) * (right_x - left_x) - (bot_y - top_y) * (X_m - right_x) <= 0] = 1
    if np.random.randint(2) == 1:
        random_bright = .4 + .6 * np.random.rand()
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image


def random_translate(image, steering_angle, trans_range):
    rows, cols, _ = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steering_angle = steering_angle + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image, steering_angle


def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype=np.float64)
    brightness_scale = .4 + np.random.uniform()
    # Scale S-channel
    image[:, :, 2] = image[:, :, 2] * brightness_scale
    # Cap S-channel values to 255
    image[:, :, 2][image[:, :, 2] > 255] = 255
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    # image = crop_img(image)
    image = rgb2yuv(image)
    return image


def augmented_img(image, steering_angle):
    image = random_shadow(image)
    # image = random_brightness(image)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, 100)
    # image = random_shadow(image)  # this here gives bad result
    image = random_brightness(image)
    return image, steering_angle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []

            for batch_sample in batch_samples:

                image, steering_angle = img_choice(batch_sample)
                image, steering_angle = augmented_img(image, steering_angle)
                # image = preprocess(image)  # Try 1 RGB 2 YUV
                images.append(image)
                steering_angles.append(steering_angle)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print('Length of train_samples: ', len(train_samples))
print('Length of validation_samples: ', len(validation_samples))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D, Conv2D, LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU

'''
ELU
'''
# model = Sequential()
# model.add(Cropping2D(cropping=((71, 25), (0, 0)), input_shape=input_shape))
# model.add(Lambda(resize_img))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), name='Conv2D_1'))
# model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2), name='Conv2D_2'))
# model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2), name='Conv2D_3'))
# model.add(Conv2D(64, (3, 3), activation="elu", name='Conv2D_4'))
# model.add(Conv2D(64, (3, 3), activation="elu", name='Conv2D_5'))
# model.add(Flatten())
# model.add(Dense(640, activation='elu', name='Dense_1'))
# model.add(Dropout(0.5, name='Dropout_1'))
# model.add(Dense(320, activation='elu', name='Dense_2'))
# model.add(Dropout(0.5, name='Dropout_2'))
# model.add(Dense(160, activation='elu', name='Dense_3'))
# model.add(Dropout(0.5, name='Dropout_3'))
# model.add(Dense(80, activation='elu', name='Dense_4'))
# model.add(Dropout(0.5, name='Dropout_4'))
# model.add(Dense(1, name='Dense_5'))
# model.summary()
#
# model.compile(loss='mse', optimizer='adam')

'''
relu
'''
model = Sequential()
model.add(Cropping2D(cropping=((71, 25), (0, 0)), input_shape=input_shape))
model.add(Lambda(resize_img))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2), name='Conv2D_1'))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2), name='Conv2D_2'))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2), name='Conv2D_3'))
model.add(Conv2D(64, (3, 3), activation="relu", name='Conv2D_4'))
model.add(Conv2D(64, (3, 3), activation="relu", name='Conv2D_5'))
model.add(Flatten())
model.add(Dense(640, activation='relu', name='Dense_1'))
model.add(Dropout(0.5, name='Dropout_1'))
model.add(Dense(320, activation='relu', name='Dense_2'))
model.add(Dropout(0.5, name='Dropout_2'))
model.add(Dense(160, activation='relu', name='Dense_3'))
model.add(Dropout(0.5, name='Dropout_3'))
model.add(Dense(80, activation='relu', name='Dense_4'))
model.add(Dropout(0.5, name='Dropout_4'))
model.add(Dense(1, name='Dense_5'))
model.summary()

model.compile(loss='mse', optimizer='adam')

'''
# LeakyReLU #
'''

# model = Sequential()
# model.add(Cropping2D(cropping=((71, 25), (0, 0)), input_shape=input_shape))
# model.add(Lambda(resize_img))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# model.add(Conv2D(24, (5, 5), strides=(2, 2), name='Conv2D_1'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_1'))
# model.add(Conv2D(36, (5, 5), strides=(2, 2), name='Conv2D_2'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_2'))
# model.add(Conv2D(48, (5, 5), strides=(2, 2), name='Conv2D_3'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_3'))
# model.add(Conv2D(64, (3, 3), name='Conv2D_4'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_4'))
# model.add(Conv2D(64, (3, 3), name='Conv2D_5'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_5'))
# model.add(Flatten())
# model.add(Dense(640, name='Dense_1'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_6'))
# model.add(Dropout(0.5, name='Dropout_1'))
# model.add(Dense(320, name='Dense_2'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_7'))
# model.add(Dropout(0.5, name='Dropout_2'))
# model.add(Dense(160, name='Dense_3'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_8'))
# model.add(Dropout(0.5, name='Dropout_3'))
# model.add(Dense(80, name='Dense_4'))
# model.add(LeakyReLU(alpha=0.001, name='LeakyReLU_9'))
# model.add(Dropout(0.5, name='Dropout_4'))
# model.add(Dense(1, name='Dense_5'))
# model.summary()
#
# model.compile(loss='mse', optimizer='adam')


history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(
    train_samples) / batch_size), validation_data=validation_generator,
    validation_steps=len(validation_samples), epochs=25, verbose=1)
# save model to .h5 file, including architechture, weights, loss, optimizer
model.save('model17_1.h5')
# model.load_model('model.h5') # reinstantiate model

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
