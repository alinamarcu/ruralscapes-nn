import numpy as np
import tensorflow as tf
import random as rn
import os
import scipy.io
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0, allow_soft_placement=True)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import cv2
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split

import time

from skimage.transform import resize

from models.safeuav import get_unet_MDCB_with_deconv_layers
from keras.optimizers import RMSprop
from models.losses import bce_dice_loss, dice_coeff

FACTOR = 2

original_width = 4096
original_height = 2160
epochs = 1000
batch_size = 2

NN_TYPE = 'safeuav'

input_width = int(original_width / FACTOR)
input_height = int(original_height / FACTOR)

model = get_unet_MDCB_with_deconv_layers(input_shape=(input_height, input_width, 3), init_nb=24, lr=0.0001, num_classes=12)    
model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
model.summary()

DATA_PATH = 'TBA'

OUTPUT_WEIGHTS = os.path.join('weights', NN_TYPE, 'gt_only_2seconds_standard_augmentation')
if not os.path.exists(OUTPUT_WEIGHTS):
    os.makedirs(OUTPUT_WEIGHTS)

train_files = open(os.path.join(DATA_PATH, 'train.txt')).read().splitlines()
valid_files = open(os.path.join(DATA_PATH, 'valid.txt')).read().splitlines()

ids_train_split = rn.sample(train_files, len(train_files)) 
ids_valid_split = rn.sample(valid_files, len(valid_files)) 

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

NUM_CLASSES = 12

def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                idsplit = id.split('_')
                video_name = idsplit[0] + '_' + idsplit[1]
                img = cv2.imread(os.path.join(DATA_PATH, 'rgb_frames', video_name + '_all', id))
                img = cv2.resize(img, (input_width, input_height))
                
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                
                label = np.load(os.path.join(DATA_PATH, 'labels', video_name, 'seg_' + id.replace('jpg', 'npz')))['arr_0']
                label_resized = resize(label, (input_height, input_width), mode='constant', anti_aliasing=True)

                img, multi_label = randomShiftScaleRotate(img, label_resized,
                                                shift_limit=(-0.0625, 0.0625),
                                                scale_limit=(-0.1, 0.1),
                                                rotate_limit=(-0, 0))
                img, multi_label = randomHorizontalFlip(img, multi_label)

                x_batch.append(img)
                y_batch.append(multi_label)
                    
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch:
                idsplit = id.split('_')
                video_name = idsplit[0] + '_' + idsplit[1]
                img = cv2.imread(os.path.join(DATA_PATH, 'rgb_frames', video_name + '_all', id))
                img = cv2.resize(img, (input_width, input_height))

                # Labels from Dragos
                label = np.load(os.path.join(DATA_PATH, 'labels', video_name, 'seg_' + id.replace('jpg', 'npz')))['arr_0']
                label_resized = resize(label, (input_height, input_width), mode='constant', anti_aliasing=True)
            
                x_batch.append(img)
                y_batch.append(label_resized)
            
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) 
            
            yield x_batch, y_batch

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=7,
                           verbose=1,
                           min_delta=1e-6),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=5,
                               verbose=1,
                               epsilon=1e-6),
             ModelCheckpoint(monitor='val_loss',
                             filepath=os.path.join(OUTPUT_WEIGHTS, 'EPOCH_{epoch:02d}_TRAIN_loss_{loss:.5f}_dice_{dice_coeff:.4f}_VALID_loss_{val_loss:.5f}_dice_{val_dice_coeff:.4f}.hdf5'),
                             save_best_only=False,
                             save_weights_only=True,
                             period=1),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
