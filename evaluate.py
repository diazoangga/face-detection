import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import create_model
from config_utils import Config

class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, threshold, num_test_files, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.threshold = threshold
        self.num_test_files = tf.cast(num_test_files, tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.math.greater_equal(y_pred, self.threshold), tf.bool)

        values = tf.equal(y_true, y_pred)
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives/self.num_test_files

# def calculating_threshold(model, num_test_files):
#     acc_list, th_list = [], []
#     for thres in range(40):
#         threshold = thres/40
#         model.compile(loss="binary_crossentropy",
#                 optimizer='adam',
#                 metrics=[BinaryTruePositives(threshold, num_test_files)])
#         _, acc = model.evaluate(test_ds, verbose=0)
#         print(f"    Threshold: {threshold} --> Accuracy: {acc}")
#         acc_list.append(acc)
#         th_list.append(threshold)

#         optimum_index = acc_list.index(max(acc_list))
#         optimum_threshold = th_list[optimum_index]

#         return acc_list, th_list, optimum_threshold


CFG = Config(config_path='./config.yml')
img_height = CFG.TRAIN.IMG_HEIGHT
img_width = CFG.TRAIN.IMG_WIDTH
weight_file = CFG.TEST.WEIGHT_FILE
test_dataset = CFG.TEST.TEST_FILE
batch_size = CFG.TEST.TEST_BATCH_SIZE
eval_result_path = CFG.TEST.EVAL_RESULT_DIR
calculate_threshold = CFG.TEST.CALCULATE_THRESHOLD
threshold = CFG.TEST.THRESHOLD

os.makedirs(eval_result_path, exist_ok=True)

print("Reading the test datasets...")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dataset,
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=[img_height, img_width],
    shuffle=True,
    seed=40,
    validation_split=0.99,
    subset='validation')

num_test_ds = len(list(test_ds))
print(f"Reading the test datasets: {num_test_ds} files is completed")

print(f"\nLoad model from {weight_file}")
model = create_model(img_height, img_width)
model.load_weights(weight_file)

print("\nEvaluating model...")

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=[BinaryTruePositives(threshold, num_test_ds)])

model.evaluate(test_ds)

if calculate_threshold == True:
    print("\nCalculating the optimum value of threshold...")
    # acc_list, th_list, optimum_threshold = calculating_threshold(model, num_test_ds)

    acc_list, th_list = [], []
    for thres in range(40):
        threshold = thres/40
        model.compile(loss="binary_crossentropy",
                optimizer='adam',
                metrics=[BinaryTruePositives(threshold, num_test_ds)])
        _, acc = model.evaluate(test_ds, verbose=0)
        print(f"    Threshold: {threshold} --> Accuracy: {acc}")
        acc_list.append(acc)
        th_list.append(threshold)

        optimum_index = acc_list.index(max(acc_list))
        optimum_threshold = th_list[optimum_index]

    print(f"Optimum threshold: {optimum_threshold}")
    plt.plot(th_list, acc_list)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(eval_result_path, 'Threshold-Accuracy_Graph.png'))
    print(f"Saving the threshold-accuracy graph is completed: {eval_result_path}")

