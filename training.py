import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import datetime
import numpy as np

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN

from model import create_model
from config_utils import Config


def lr_function(initial_lr, final_lr, num_epochs, batch_size, num_train_ds):
    learning_rate_decay_factor = (final_lr / initial_lr)**(1/num_epochs)
    steps_per_epoch = int(num_train_ds/batch_size)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=steps_per_epoch,
                    decay_rate=learning_rate_decay_factor,
                    staircase=False)
    return lr_schedule

tf.random.set_seed(40)

CFG = Config(config_path='./config.yml')
img_height = CFG.TRAIN.IMG_HEIGHT
img_width = CFG.TRAIN.IMG_WIDTH
batch_size = CFG.TRAIN.BATCH_SIZE
initial_lr = CFG.SOLVER.INITIAL_LR
final_lr = CFG.SOLVER.FINAL_LR
num_epochs = CFG.TRAIN.EPOCH_NUMS
dataset_path = CFG.DATASET.DATASET_DIR
testing_path = CFG.TEST.TEST_FILE
saved_path = CFG.TRAIN.MODEL_SAVE_DIR
log_path = CFG.TRAIN.TBOARD_SAVE_DIR

os.makedirs(saved_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)


## READ DATASETS
print("Reading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=[img_height, img_width],
    shuffle=True,
    seed=40,
    validation_split=0.15,
    subset='training')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=[img_height, img_width],
    shuffle=True,
    seed=40,
    validation_split=0.15,
    subset='validation')

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # testing_path,
    # labels='inferred',
    # label_mode='binary',
    # color_mode='rgb',
    # batch_size=1,
    # image_size=[img_height, img_width],
    # shuffle=True,
    # seed=40,
    # validation_split=0.99,
    # subset='validation')

num_train_ds = len(list(train_ds))
num_val_ds = len(list(val_ds))
# num_test_ds = len(list(test_ds))

print("Reading datasets is completed with:")
print(f"    train datasets: {num_train_ds} files")
print(f"    validation datasets: {num_val_ds} files")
# print(f"    test datasets: {num_test_ds} files")


## BUILD THE MODEL
print("Build the model...")
model = create_model(img_height, img_width)
print("The model is sucessfully built...")
model.summary()

## COMPILE THE MODEL
print("Compiling the model...")
lr_schedule = lr_function(initial_lr, final_lr, num_epochs, batch_size, num_train_ds)
checkpoint = ModelCheckpoint(filepath=os.path.join(saved_path, 'img_class.epoch{epoch:02d}-loss{loss:.2f}.h5'),
                            monitor='val_loss',
                            verbose=1,
                            save_weights_only=True,
                            save_best_only=True,
                            mode='min')
terminateNaN = TerminateOnNaN()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
log_dir = os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=["accuracy"])

print(f'''
Training model profile:
    Number of epochs    : {num_epochs}
    Batch size          : {batch_size}
    Initial LR          : {initial_lr}
    Final LR            : {final_lr}
    Optimizer           : Adam
''')

## TRAIN THE MODEL
history = model.fit(
    train_ds,
#     steps_per_epoch=8, 
    epochs=num_epochs,
    verbose=1,
    validation_data=val_ds,
    callbacks=[LearningRateScheduler(lr_schedule, verbose=1), early_stop, tensorboard_callback, checkpoint]
#     validation_steps=8
)

## SAVE THE MODELS
model.save_weights(os.path.join(saved_path, 'last_model.h5'))
np.save(f'{saved_path}/history.npy',history.history)
print(f'Training result is saved in {saved_path} for the model, and {log_dir} for the tensorboard')