from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import random
from keras.preprocessing.image import img_to_array,load_img
from keras.utils import np_utils
import os
import os.path


# 训练的图片放在train文件夹里，可自定义文件夹。
def train_generator(perfix='train', batch_size=64):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    filename = 'train.csv'
    with open(filename, 'r') as f:
        all_lines = []
        for line in f:
            all_lines.append(line)
        all_num_img = len(all_lines)

        while True:
            random.shuffle(all_lines)
            i = 0
            while i < all_num_img:
                begin = i
                end = begin + batch_size
                if end > all_num_img:
                    end = all_num_img
                x, y1, y2, y3, y4, y5 = get_batch_imgs_labels(perfix, all_lines, begin, end)
                i += batch_size

                train_datagen.fit(x)
                yield (x, [y1, y2, y3, y4, y5])



# 验证集的图片放在test文件夹里，可自定义文件夹。
def val_generator(perfix='test', batch_size=16):

    val_datagen = ImageDataGenerator(rescale=1./255)

    filename = 'test.csv'
    with open(filename, 'r') as f:
        all_lines = []
        for line in f:
            all_lines.append(line)
        all_num_img = len(all_lines)

        while True:
            random.shuffle(all_lines)
            i = 0
            while i < all_num_img:
                begin = i
                end = begin + batch_size
                if end > all_num_img:
                    end = all_num_img
                x, y1, y2, y3, y4, y5 = get_batch_imgs_labels(perfix, all_lines, begin, end)
                i += batch_size

                val_datagen.fit(x)
                yield (x, [y1, y2, y3, y4, y5])



def get_batch_imgs_labels(perfix, all_lines, begin, end):
    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    for line in all_lines[begin: end]:
        totakes = line.split(',')
        img_path = os.path.join(perfix, totakes[0])
        img = load_img(img_path, target_size=(299, 299))
        img = img_to_array(img)
        x.append(np.array(img))
        y1.append(int(totakes[1]))
        y2.append(int(totakes[1]))
        y3.append(int(totakes[1]))
        y4.append(int(totakes[1]))
        y5.append(int(totakes[1]))

    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    y5 = np.array(y5)
    
    #40为每个输出的类别数，可自定义
    return x, np_utils.to_categorical(y1, 40), \
           np_utils.to_categorical(y2, 40), \
           np_utils.to_categorical(y3, 40), \
           np_utils.to_categorical(y4, 40), \
           np_utils.to_categorical(y5, 40)





