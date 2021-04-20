import sys
import os
import itertools
import random
from PIL import Image
from libsvm.svmutil import *
import pandas as pd

ROOT_DIR = './flowers/'
FLOWERS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
DIMENSION = 32

def get_data():
    # build image list
    data = {}
    desc = {}
    i = 0.0
    for flower_name in FLOWERS:
        dir_name = ROOT_DIR + flower_name + '/'
        imgs = [Image.open(dir_name + file_name).resize((DIMENSION, DIMENSION)) for file_name in os.listdir(dir_name)]
        imgs = [list(itertools.chain.from_iterable(img.getdata())) + [i] for img in imgs]
        data[flower_name] = imgs
        desc[flower_name] = i
        i += 1

    list_2d_vector = []
    for key in data.keys():
        for vec in data[key]:
            list_2d_vector.append(vec)

    return desc, list_2d_vector



def main():
    desc, vecs = get_data()
    print('Length:', len(vecs)) #1304
    print('Type:', type(vecs)) #1304
    print('Description:', desc)
    print('======================================')
    print(vecs[0])

    dataset = pd.DataFrame(vecs)
    dataset.to_csv('images_dataset_vector.csv')
    with open('./description.txt', 'w') as f:
        f.write(str(desc))

    return

main()