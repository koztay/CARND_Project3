import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # this line makes the header line ignored
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # greater corection factor tends to track to go right, lesser tends to go left...
            # correction = 0.254  # with augmented data without track 2 data very hard to tune!!!
            # correction =0.253 # with augmented data without track 2 data very hard to tune!!!
            # correction = 0.2535 # with augmented data without track 2 data very hard to tune!!!
            # correction = 0.32 without augmented data without track2 data

            correction = 0.2  # with augmented data + with track2 data succesful driving for both tracks
            # add center images and angles in the bacth

            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # add left images in the batch and add corection angles
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)

            # add rigth images in the batch and add corecction angles
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)

            # augmenting images
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):  # zipping the arrays from the above
                augmented_images.append(image)
                augmented_angles.append(angle)
                bgr_image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # used rgb images for augmented images
                augmented_images.append(cv2.flip(rgb_image, 1))
                augmented_angles.append(angle * -1.0)  # flipped, reverse direction so multiplied by -1.0

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# model = Sequential()
model = load_model('weights.best.h5')

# # # # checkpoint
filepath = "weights.best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

samples_per_epoch = len(train_samples)
nb_val_samples = len(validation_samples)

model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    nb_epoch=10,
    callbacks=callbacks_list,
)

model.save('model.h5')
