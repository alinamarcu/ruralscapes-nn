import numpy as np
import tensorflow as tf
import random as rn
from tqdm import tqdm
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

import time

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from natsort import natsorted

import cv2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler

FACTOR = 2

NUM_CLASSES = 12

#labels = {'land', 'forest', 'residential', 'capitze', 'road', 'church', 'cars', 'water', 'sky', 'hill', 'person', 'fence'};
CLASSES_BGR = ((0, 255, 0), (0, 127, 0), (0, 255, 255), (0, 127, 255), (255, 255, 255), (255, 0, 255), (127, 127, 127), (255, 0, 0), (255, 255, 0), (63, 127, 127), (0, 0, 255), (0, 127, 127))

from models.safeuav import get_unet_MDCB_with_deconv_layers

original_width = 4096
original_height = 2160

input_width = int(original_width / FACTOR)
input_height = int(original_height / FACTOR)

batch_size = 1

DATA_PATH = './data/rgb_frames'

# baseline
#WEIGHTS_PATH = './weights/safeuav/gt_only_2seconds_standard_augmentation/EPOCH_153_TRAIN_loss_0.21953_dice_0.8511_VALID_loss_0.34358_dice_0.7740.hdf5' 

# segprop - with denoise
#WEIGHTS_PATH = './weights/safeuav/segprop_interp_all_standard_augmentation/with_denoise/EPOCH_13_TRAIN_loss_0.03721_dice_0.9745_VALID_loss_0.24476_dice_0.8702.hdf5'

# segprop - without denoise
WEIGHTS_PATH = './weights/safeuav/segprop_interp_all_standard_augmentation/without_denoise/EPOCH_18_TRAIN_loss_0.03984_dice_0.9727_VALID_loss_0.24240_dice_0.8683.hdf5'

# LOAD MODEL
model = get_unet_MDCB_with_deconv_layers(input_shape=(input_height, input_width, 3), init_nb=24, lr=0.0001, num_classes=NUM_CLASSES)
model.summary()
model.load_weights(WEIGHTS_PATH)

testing_clips = [x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))]

for pred_video_id in testing_clips:
    pred_video_id = pred_video_id.split('.')[0]
    print('Predicting on clip: ' + pred_video_id)
    test_files = natsorted([x for x in os.listdir(os.path.join(DATA_PATH, pred_video_id)) if x.endswith('.png')])
    ids_test_split = rn.sample(test_files, len(test_files))
    
    # RESULTS DIR
    BEST_EPOCH = os.path.basename(WEIGHTS_PATH).split('_')[1]
    results_dir = os.path.join(os.path.dirname(WEIGHTS_PATH).replace('/weights/', '/results/'), 'best_epoch_' + BEST_EPOCH, pred_video_id, 'overlap')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print('Predicting on {} samples with batch_size = {}...'.format(len(test_files), batch_size))
    for start in tqdm(range(0, len(test_files), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(ids_test_split))
        ids_test_batch = ids_test_split[start:end]
        for current_file in ids_test_batch:
            orig_img = cv2.imread(os.path.join(DATA_PATH, pred_video_id, current_file))
            orig_height, orig_width, _ = orig_img.shape
            
            input_img = cv2.resize(orig_img, (input_width, input_height))
            
            x_batch.append(input_img)
            
        x_batch = np.array(x_batch, np.float32) / 255
        
        preds = model.predict_on_batch(x_batch)
        
        output_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        
        for idx, pred in enumerate(preds):
            _,_,num_classes = pred.shape
            output_label = cv2.resize(pred, (input_width, input_height))
            label_indices = output_label.argmax(axis=2)
            
            for current_prediction_idx in range(NUM_CLASSES):
                output_image[np.nonzero(np.equal(label_indices,current_prediction_idx))] = CLASSES_BGR[current_prediction_idx]
            
            output_image = cv2.resize(output_image, (orig_width, orig_height))
            alpha = 0.4
            cv2.addWeighted(output_image, alpha, orig_img, 1 - alpha, 0, output_image)
            cv2.imwrite(os.path.join(results_dir, os.path.basename(current_file)), output_image)

        
