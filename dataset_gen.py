import json
import os
import os.path as ops
import cv2
import random
import glob

from config_utils import Config

def extract_pos(dataset_path, info_json):
    """
    Extract positive samples from the given dataset.
    Input:
        info_json       : Extracted information from json
        dataset_path    : Dataset path
    Output:
        -
    """
    print('\nExtracting positive samples from dataset...')
    save_folder = ops.join(dataset_path, 'images','extracted_img','positive')
    os.makedirs(save_folder, exist_ok=True)
    print("Folder for extracted images (positive) is ready!")
    for image_name in info_json:
        image_path = ops.join(dataset_path,'images',image_name)
        
        for idx, bbox in enumerate(info_json[image_name]):
            save_name = ops.join(save_folder,f'{image_name[:-4]}_{idx}.jpg')
            uncropped_img = cv2.imread(image_path)
            y_min = bbox[1]
            y_max = bbox[3]
            x_min = bbox[0]
            x_max = bbox[2]
            cropped_img = uncropped_img[y_min:y_max, x_min:x_max]
            assert((y_max < uncropped_img.shape[0]) and (x_max < uncropped_img.shape[1])), f'{x_max}  {y_max}'
            cv2.imwrite(save_name, cropped_img)
            print(f'Cropped image is saved successfully: {save_name}')


def extract_neg(dataset_path, info_json, num_files):
    """
    Extract negative samples from the given dataset.
    Input:
        info_json       : Extracted information from json
        dataset_path    : Dataset path
        num_files   : Number of files inside the given dataset
    Output:
        -
    """
    print('\nGenerating random negative samples from dataset...')
    save_folder = ops.join(dataset_path, 'images','extracted_img','negative')
    os.makedirs(save_folder, exist_ok=True)
    print("Folder for extracted images (positive) is ready!")
    count = 0
    while count < num_files:
        image_name = f'{str(random.randint(0,830)).zfill(4)}.jpg'
        save_name = ops.join(save_folder, f'{count}.jpg')
        image_path = ops.join(dataset_path,'images',image_name)
        inside = False      
        
        uncropped_img = cv2.imread(image_path)
        scale = 0.2
        height, width = int(uncropped_img.shape[0]*scale), int(uncropped_img.shape[1]*scale)
        x1 = random.randint(0, uncropped_img.shape[1] - width)
        y1 = random.randint(0, uncropped_img.shape[0] - height)
        x2, y2 = x1+width, y1+height
        for bbox in info_json[image_name]:
            
            y_min = bbox[1]
            y_max = bbox[3]
            x_min = bbox[0]
            x_max = bbox[2]
            

            if x_min<=x2 and x_max>=x1 and y_min<=y2 and y_max>=y1: #if the random samples is inside the bounding box
                inside=True

        
        if inside == False: #if the random samples is not inside the bounding box, save the img
            cropped_img = uncropped_img[y1:y2, x1:x2]
            # print(uncropped_img.shape)
            # print(x1,x2,y1,y2)
            # print(x_min,x_max,y_min,y_max)
            # print(cropped_img.shape)
            cv2.imwrite(save_name, cropped_img)
            print(f'Cropped image is saved successfully: {save_name}')
            count+=1
            
        else:           #if the random samples is not inside the bounding box, skip
            print('Skip')

def data_augmentation(dataset_path, rot_angle=30):
    """
    Data augmentation with flipping horizontal and vertical and rotation
    """
    for img_path in glob.glob(f'{dataset_path}/*/*.jpg'):
        img = cv2.imread(img_path)
        flip_img = cv2.flip(img,1)
        flip_vertical_img = cv2.flip(img,0)
        angle = int(random.uniform(-1*rot_angle, rot_angle))
        height, width = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(height/2), int(width/2)), angle, 1)
        rot_img = cv2.warpAffine(img, M, (width, height))
        cv2.imwrite(f'{img_path[:-4]}_flip.jpg', flip_img)
        cv2.imwrite(f'{img_path[:-4]}_flip_vertical.jpg', flip_vertical_img)
        cv2.imwrite(f'{img_path[:-4]}_rot.jpg', rot_img)


CFG = Config(config_path='./config.yml')
raw_dataset_path = str(CFG.DATASET.RAW_DATA_DIR)
clean_dataset_path = CFG.DATASET.DATASET_DIR
num_files = CFG.DATASET.POS_DATASET_NUM

json_file = open(ops.join(raw_dataset_path,'annotations.json'))
images_json = json.load(json_file)

extract_pos(raw_dataset_path, images_json)
extract_neg(raw_dataset_path, images_json, num_files)

data_augmentation(clean_dataset_path)

num_pos_files = len(glob.glob(f'{clean_dataset_path}/positive/*'))
num_neg_files = len(glob.glob(f'{clean_dataset_path}/negative/*'))

print('\nDataset preparation is completed succesfully with:')
print(f'    Number of positive files: {num_pos_files}')
print(f'    Number of negative files: {num_neg_files}')

