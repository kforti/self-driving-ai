import shutil
import os
from pathlib import Path

import cv2
import pandas as pd
from skimage.io import imread, imsave


def extract_data(log_path, save_log, save_img):
    if not os.path.exists(save_img):
        os.mkdir(save_img)

    df = pd.read_csv(log_path, header=None)
    fdf = pd.DataFrame()
    # We only need the center image, the steering angle and the throttle
    fdf[0] = df[0]
    fdf[1] = df[3]
    fdf[2] = df[4]

    for row in df.iterrows():
        path = row[1].values[0]
        new_path = os.path.join(save_img, Path(path).name)
        #t = fdf.iloc[0, row[0]]

        fdf.iloc[row[0], 0] = new_path

        if not os.path.exists(new_path):
            shutil.move(path, new_path)

    if os.path.exists(save_log):
        old_df = pd.read_csv(save_log, header=None)
        fdf = old_df.append(fdf)
    fdf.to_csv(save_log, index=False, header=False)


def flip_image(img, steering_angle):
    steering_angle = -steering_angle
    img = cv2.flip( img, 1 )
    return img, steering_angle

if __name__ == '__main__':

    ADD_LOG_PATH = "/home/kevin/Desktop/driving_log.csv"
    ADD_IMG_PATH = "/home/kevin/Downloads/IMG"

    IMG_PATH = "../Data/Training_Images"
    LOG_PATH = "../Data/Training_Data"

    with open(LOG_PATH, "r") as f:
        for row in f:
            path, angle, throttle = row.split(",")
            img = imread(path)
            flipped_img, flipped_angle = flip_image(img, float(angle))
            print(angle)
            imsave("img.jpeg", img)
            imsave("flipped_img.jpeg", flipped_img)
            print(flipped_angle)
            break
    #extract_data(ADD_LOG_PATH, LOG_PATH,IMG_PATH)
