# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:32:37 2020

@author: Guilherme
"""

import cv2
from PIL import Image
import numpy as np

def get_frames_from_video(vid, frames = 'All'):
    video_capture = cv2.VideoCapture(vid)
    v_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    if frames == 'All':
        frames_to_get = range(v_len)
    else:
        if type(frames) != list:
            raise Exception('Need list or "all" as second parameter (frames to return)')
        frames_to_get = frames
    for j in range(v_len):
        success, vframe = video_capture.read()
        if j in frames_to_get:
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            all_frames = all_frames + [vframe]
        if j > max(frames_to_get):
            break
    return all_frames

def shave_black_bit(img, thr = 0.05):
    img_a = np.array(img)
    img_superposed = img_a[:, :, 0] + img_a[:, :, 1] + img_a[:, :, 2]
    nr_rows, nr_cols = img_superposed.shape
    
    black_rows = [sum(row != 0) for row in img_superposed]
    black_cols = [sum(row != 0) for row in img_superposed.transpose()]
    
    thr_r = int(nr_rows * thr)
    thr_c = int(nr_cols * thr)
    
    under_thr_row = [i for i in range(nr_rows) if black_rows[i] < thr_r]
    under_thr_col = [i for i in range(nr_cols) if black_cols[i] < thr_c]
    
    try:
        new_img_start_r = min((set(range(nr_rows)) - set(under_thr_row)))
    except:
        new_img_start_r = 0
    try:
        new_img_end_r = max((set(range(nr_rows)) - set(under_thr_row)))
    except:
        new_img_end_r = nr_rows
    if new_img_start_r == new_img_end_r:
        new_img_end_r = new_img_end_r + 1

    try:
        new_img_start_c = min((set(range(nr_rows)) - set(under_thr_col)))
    except:
        new_img_start_c = 0
    try:
        new_img_end_c = max((set(range(nr_cols)) - set(under_thr_col)))
    except:
        new_img_end_c = nr_cols
    if new_img_start_c == new_img_end_c:
        new_img_end_c = new_img_end_c + 1
        
    return Image.fromarray(img_a[new_img_start_r:new_img_end_r, new_img_start_c:new_img_end_c, :])
