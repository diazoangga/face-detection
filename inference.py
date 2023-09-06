import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import PIL.Image as Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

from model import create_model
from config_utils import Config

def preprocess_img(image, img_resize):
    img = Image.open(image)
    img = img.resize(img_resize)
    img = tf.expand_dims(img, axis=0)
    return img


CFG = Config(config_path='./config.yml')
img_height = CFG.TRAIN.IMG_HEIGHT
img_width = CFG.TRAIN.IMG_WIDTH
threshold = CFG.TEST.THRESHOLD
weight_file = CFG.TEST.WEIGHT_FILE
test_dataset = CFG.TEST.TEST_FILE
eval_result_path = CFG.TEST.EVAL_RESULT_DIR
inference_mode = CFG.INFERENCE.MODE


if inference_mode=='DATASET':    
    print(f"\nLoad model from {weight_file}...")
    model = create_model(img_height, img_width)
    model.load_weights(weight_file)
    data = dict()

    with open(f'./{eval_result_path}/test_dataset_result.json', 'w') as file:
        for image_path in glob.glob(f"{test_dataset}/*/*.jpg"):
            img = preprocess_img(image_path, [img_height, img_width])
            confidence_level = model(img)
            class_result = int(confidence_level >= threshold)
            
            data[os.path.split(image_path)[-1]] = [class_result, float(np.array(confidence_level)[0][0])]

        # print(data)

        json.dump(data, file, sort_keys=True)

    print(f'Resulted JSON is saved on ./{eval_result_path}/test_dataset_result.json')

elif inference_mode=='IMAGE':
    img_file = CFG.INFERENCE.IMG_FILE
    print(f"\nLoad model from {weight_file}...")
    model = create_model(img_height, img_width)
    model.load_weights(weight_file)
    img = preprocess_img(img_file, [img_height, img_width])
    confidence_level = model(img)
    class_result = 'face' if (confidence_level >= threshold) else 'non_face'

    print(f'''
    Result of the classification:
        Confidence level    : {np.array(confidence_level)[0][0]*100} %
        Class               : {class_result}
        ''')
    
else:
    print('Please type between "DATASET" or "IMAGE" on Config file')