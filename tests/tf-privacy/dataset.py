import tensorflow as tf
import numpy as np
import tarfile
import urllib
import os

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227, 227))
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


def load_texas100():
    base_path = '../dataset/dataset_texas/'
    features = os.path.join(base_path,'texas/100/feats')
    labels = os.path.join(base_path,'texas/100/labels')        
    
    if not os.path.isfile(features):
        print("Dowloading the dataset...")
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",os.path.join(base_path,'tmp.tgz'))
        print('Dataset Dowloaded')

        tar = tarfile.open(os.path.join(base_path,'tmp.tgz'))
        tar.extractall(path=base_path)
    
    
    data_set_features =np.genfromtxt(features,delimiter=',')
    data_set_label =np.genfromtxt(labels,delimiter=',')


    X = data_set_features.astype(np.float64)
    Y = data_set_label.astype(np.int32)-1
    
    len_train =len(X)
    r = np.load(base_path+'random_r_texas100.npy')
    X=X[r]
    Y=Y[r]
    train_classifier_ratio, train_attack_ratio = float(10000)/float(X.shape[0]),0.3
    train_target_data = X[:int(train_classifier_ratio*len_train)]
    train_attack_data = X[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    train_target_label = Y[:int(train_classifier_ratio*len_train)]
    train_attack_label = Y[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    print('TRain len', len_train)
    print(train_target_data.shape)
    print(train_target_label.shape)
    print(train_attack_data.shape)
    print(train_attack_label.shape)
    print(test_data.shape)
    print(test_label.shape)    
   
    return  test_data, test_label, train_target_data, train_target_label, train_attack_data, train_attack_label

def load_reuters():
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.reuters.load_data(num_words=None, test_split=0.2)
    word_index = tf.keras.datasets.reuters.get_word_index(path='reuters_word_index.json')
    index_to_word = {}
    for key, val in word_index.items():
        index_to_word[val] = key
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    x_train = tokenizer.sequences_to_matrix(train_data, mode='binary')
    x_test = tokenizer.sequences_to_matrix(test_data, mode='binary')
    # y_train = tf.keras.utils.to_categorical(train_labels, n_classes)
    # y_test = tf.keras.utils.to_categorical(test_labels, n_classes)
    # print(x_train.shape, train_labels.shape, test_labels.shape)
    return x_train, train_labels, x_test, test_labels

    

#load_texas100()
#load_reuters()
