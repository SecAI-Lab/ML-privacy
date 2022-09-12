import tensorflow as tf


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (224, 224))
    return image, label


def load_cifar100():
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.cifar100.load_data()

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    train_data = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=100, drop_remainder=False))
    test_data = (test_ds
                 .map(process_images)
                 .shuffle(buffer_size=test_ds_size)
                 .batch(batch_size=10, drop_remainder=False))

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels
