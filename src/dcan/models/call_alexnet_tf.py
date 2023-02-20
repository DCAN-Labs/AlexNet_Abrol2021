# 1. Tools And Libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

# 2. Dataset
from dcan.models.alex_net import get_alex_net

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

# 3. Preprocessing
plt.figure(figsize=(20, 20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')


def process_images(image_, label_):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image_ = tf.image.per_image_standardization(image_)
    # Resize images from 32x32 to 277x277
    image_ = tf.image.resize(image_, (227, 227))
    return image_, label_


# 4. Data/Input Pipeline
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds
           .map(process_images)
           .shuffle(buffer_size=train_ds_size)
           .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
                 .map(process_images)
                 .shuffle(buffer_size=train_ds_size)
                 .batch(batch_size=32, drop_remainder=True))

# 5. Model Implementation
model = get_alex_net()
# 6. TensorBoard
root_logdir = os.path.join(os.curdir, "logs\\fit\\")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# 7. Training and Results
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(train_ds,
          epochs=50,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])
