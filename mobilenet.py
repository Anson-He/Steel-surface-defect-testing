from keras import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNet
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from time import *

cla = os.listdir('./train')
train_path = './train'
val_path = './val'
test_path = './test'
train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # 归一化
    rotation_range=40,  # 旋转
    width_shift_range=0.2,  # 水平偏移
    height_shift_range=0.2,  # 垂直偏移
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    horizontal_flip=True  # 水平翻转
)

val_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator = train_data.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical',
    save_to_dir='./new',
    save_format='jpg'
)

val_generator = val_data.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)

test_generator = test_data.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)


def mobilenet(train_generator, val_generator, save_model_path = './model/'+'mobilenet.h5', log_dir = './logs/mobilenet'):
    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.summary()
    plot_model(model, to_file='./mobilenet/mobilenet_bottleneck.png', show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint = ModelCheckpoint(filepath=save_model_path, monitor='acc', mode='auto', save_best_only='True')

    model.fit_generator(
        generator=train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[tensorboard, checkpoint],
    )
    model.save(save_model_path)
    return model


def evaluate_model(test_data, save_model_path):
    model = tf.keras.models.load_model(save_model_path)
    loss, accuracy = model.evaluate_generator(test_data)
    print('\n Loss:%.2f,Accuracy:%.2f%%' % (loss, accuracy*100))
    s = 0
    for i in range(len(test_data)):
        image, label = test_data.next()
        begin = time()
        predict = np.argmax(model.predict(image))
        end = time()
        runtime = end - begin
        s = s + runtime
        # plt.figure()
        # plt.imshow(np.squeeze(image))
        # plt.title(cla[predict])
        # plt.show()
    print('平均运行时间为:', s / len(test_data))

# mobilenet(train_generator, val_generator)
evaluate_model(test_generator, './model/mobilenet.h5')
