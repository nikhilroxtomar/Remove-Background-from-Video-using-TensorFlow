
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("processed_videos")
    create_dir("frames")

    """ Loading model: DeepLabV3+ """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")
    # model.summary()

    """ Video Path """
    video_path = "videos/Elon_Musk.mp4"

    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    h, w, _ = frame.shape
    vs.release()

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(f'processed_videos/Elon_Musk.avi', fourcc, 30, (w, h), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break

        h, w, _ = frame.shape
        ori_frame = frame
        frame = cv2.resize(frame, (W, H))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0

        mask = model.predict(frame)[0]
        mask = cv2.resize(mask, (w, h))
        mask = mask > 0.5
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        photo_mask = mask
        background_mask = np.abs(1-mask)

        masked_frame = ori_frame * photo_mask

        background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
        background_mask = background_mask * [0, 0, 255]
        final_frame = masked_frame + background_mask
        final_frame = final_frame.astype(np.uint8)

        cv2.imwrite(f"frames/{idx}.png", final_frame)
        idx += 1

        out.write(final_frame)
