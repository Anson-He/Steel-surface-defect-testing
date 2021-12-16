import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

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
    batch_size=8,
    class_mode='categorical',
    # save_to_dir='./new',
    save_format='jpg'
)

val_generator = val_data.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)

test_generator = val_data.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)


def model(train_generator, val_generator, save_model_path = './model/'+'self_CNN.h5', log_dir = './logs'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(224, 224, 3),
                                     kernel_size=(5, 5),
                                     filters=3,
                                     activation=tf.nn.relu))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3),
                                     filters=256,
                                     activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3),
                                     filters=128,
                                     activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit_generator(
        generator=train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=[tensorboard],
    )
    model.save(save_model_path)
    return model


def evaluate_model(test_data, save_model_path):
    model = tf.keras.models.load_model(save_model_path)
    loss, accuracy = model.evaluate_generator(test_data)
    print('\n Loss:%.2f,Accuracy:%.2f%%' % (loss, accuracy*100))
    for i in range(6):
        image, label = test_data.next()
        predict = np.argmax(model.predict(image))
        plt.figure()
        plt.imshow(np.squeeze(image))
        plt.title(cla[predict])
        plt.show()


# model(train_generator, val_generator)
evaluate_model(test_generator, './model/self_CNN.h5')
