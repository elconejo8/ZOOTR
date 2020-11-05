# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:48:03 2020

@author: Guilherme
"""
import os
from functions import shave_black_bit
from PIL import Image


def build_processed_images(input_folder, output_folder):
    for i in os.listdir(input_folder):
        img = Image.open(input_folder + i)
        new_img = shave_black_bit(img)
        new_img.save(output_folder + i)


build_processed_images('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Random_Frames\\', 'C:\\Users\\Guilherme\\Documents\\ZOOTR\\Processed\\Random_Frames\\')
build_processed_images('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Items\\', 'C:\\Users\\Guilherme\\Documents\\ZOOTR\\Items\\')
build_processed_images('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Test\\Random_Frames\\', 'C:\\Users\\Guilherme\\Documents\\ZOOTR\\Processed\\Test\\Random_Frames\\')
build_processed_images('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Test\\Items\\', 'C:\\Users\\Guilherme\\Documents\\ZOOTR\\Processed\\Test\\Items\\')

