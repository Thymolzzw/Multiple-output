from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adagrad, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
from keras.utils import np_utils
import os.path
from read import train_generator

# 共五个输出，每个输出的类别个数为num_output
num_output = [40, 40, 40, 40, 40]
tensorboard = TensorBoard(log_dir=os.path.join('logs'))
checkpoint=ModelCheckpoint(
    #  filepath=os.path.join('checkpoint', 'inceptionV3.{epoch:03d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
        filepath=os.path.join('checkpoint', 'inceptionV3.{epoch:03d}-{out1_acc:.3f}-{out2_acc:.3f}-{out3_acc:.3f}-{out4_acc:.3f}'
                                            +'-{out5_acc:.3f}.hdf5'),
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        period=1
        )

base_model = InceptionV3(weights='imagenet', include_top=False)

x1 = base_model.output
x1 = Conv2D(filters=1024,kernel_size=(3,3),use_bias=True,padding='same')(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding="same")(x1)
x1 = Dropout(rate=0.25)(x1)
#x1 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x1)
#x1 = MaxPooling2D(pool_size=(2,2),padding="same")(x1)
#x1 = Dropout(rate=0.25)(x1)
x1 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding="same")(x1)
x1 = Dropout(rate=0.25)(x1)
x1 = Conv2D(filters=1024,kernel_size=(3,3),use_bias=True,padding="same")(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding="same")(x1)
x1 = Dropout(rate=0.25)(x1)
x1 = GlobalAveragePooling2D()(x1)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x1 = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x1)  # 1024
x1 = Dropout(rate=0.5)(x1)
out1 = Dense(num_output[0], activation='softmax', name='out1')(x1)

x2 = base_model.output
x2 = Conv2D(filters=1024,kernel_size=(3,3), use_bias=True,padding="same")(x2)
x2 = MaxPooling2D(pool_size=(2, 2), padding="same")(x2)
x2 = Dropout(rate=0.25)(x2)
#x2 = Conv2D(filters=2048, kernel_size=(3, 3),use_bias=True, padding="same")(x2)
#x2 = MaxPooling2D(pool_size=(2,2), padding="same")(x2)
#x2 = Dropout(rate=0.25)(x2)
x2 = Conv2D(filters=2048, kernel_size=(3, 3),use_bias=True, padding="same")(x2)
x2 = MaxPooling2D(pool_size=(2,2), padding="same")(x2)
x2 = Dropout(rate=0.25)(x2)
x2 = Conv2D(filters=1024,kernel_size=(3,3), use_bias=True,padding="same")(x2)
x2 = MaxPooling2D(pool_size=(2, 2), padding="same")(x2)
x2 = Dropout(rate=0.25)(x2)
x2 = GlobalAveragePooling2D()(x2)
x2 = Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.001))(x2)
x2 = Dropout(rate=0.5)(x2)
out2 = Dense(num_output[1], activation='softmax', name='out2')(x2)

x3 = base_model.output
x3 = Conv2D(filters=1024,kernel_size=(3,3),use_bias=True,padding="same")(x3)
x3 = MaxPooling2D(pool_size=(2,2),padding="same")(x3)
x3 = Dropout(rate=0.25)(x3)
#x3 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x3)
#x3 = MaxPooling2D(pool_size=(2,2),padding="same")(x3)
#x3 = Dropout(rate=0.25)(x3)
x3 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x3)
x3 = MaxPooling2D(pool_size=(2,2),padding="same")(x3)
x3 = Dropout(rate=0.25)(x3)
x3 = Conv2D(filters=1024,kernel_size=(3,3),use_bias=True,padding="same")(x3)
x3 = MaxPooling2D(pool_size=(2,2),padding="same")(x3)
x3 = Dropout(rate=0.25)(x3)
x3 = GlobalAveragePooling2D()(x3)
x3 = Dense(units=1024,activation="relu", kernel_regularizer=regularizers.l2(0.001))(x3)
x3 = Dropout(rate=0.5)(x3)
out3 = Dense(num_output[2], activation="softmax", name='out3')(x3)

x4 = base_model.output
x4 = Conv2D(filters=1024,kernel_size=(3,3), use_bias=True,padding="same")(x4)
x4 = MaxPooling2D(pool_size=(2,2),padding="same")(x4)
x4 = Dropout(rate=0.25)(x4)
#x4 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x4)
#x4 = MaxPooling2D(pool_size=(2,2),padding="same")(x4)
#x4 = Dropout(rate=0.25)(x4)
x4 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x4)
x4 = MaxPooling2D(pool_size=(2,2),padding="same")(x4)
x4 = Dropout(rate=0.25)(x4)
x4 = Conv2D(filters=1024,kernel_size=(3,3), use_bias=True,padding="same")(x4)
x4 = MaxPooling2D(pool_size=(2,2),padding="same")(x4)
x4 = Dropout(rate=0.25)(x4)
x4 = GlobalAveragePooling2D()(x4)
x4 = Dense(units=1024,activation="relu",kernel_regularizer=regularizers.l2(0.001))(x4)
x4 = Dropout(rate=0.5)(x4)
out4 = Dense(num_output[3], activation="softmax", name='out4')(x4)

x5 = base_model.output
x5 = Conv2D(filters=1024,kernel_size=(3,3),use_bias=True,padding="same")(x5)
x5 = MaxPooling2D(pool_size=(2,2),padding="same")(x5)
x5 = Dropout(rate=0.25)(x5)
#x5 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x5)
#x5 = MaxPooling2D(pool_size=(2,2),padding="same")(x5)
#x5 = Dropout(rate=0.25)(x5)
x5 = Conv2D(filters=2048,kernel_size=(3,3),use_bias=True,padding="same")(x5)
x5 = MaxPooling2D(pool_size=(2,2),padding="same")(x5)
x5 = Dropout(rate=0.25)(x5)
x5 = Conv2D(filters=1024,kernel_size=(3,3),use_bias=True,padding="same")(x5)
x5 = MaxPooling2D(pool_size=(2,2),padding="same")(x5)
x5 = Dropout(rate=0.25)(x5)
x5 = GlobalAveragePooling2D()(x5)
x5 = Dense(units=1024,activation="relu",kernel_regularizer=regularizers.l2(0.001))(x5)
x5 = Dropout(rate=0.5)(x5)
out5 = Dense(num_output[4], activation="softmax", name='out5')(x5)

model = Model(inputs=base_model.input, outputs=[out1, out2, out3, out4, out5])

for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=SGD(lr=0.001,momentum=0.9,decay=0.0001,nesterov=True), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator=train_generator(), epochs=100, shuffle=True,
                    steps_per_epoch=230,
                    callbacks=[tensorboard, checkpoint])

model.save('final.h5')

