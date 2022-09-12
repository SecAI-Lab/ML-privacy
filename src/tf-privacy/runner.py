from report import get_report, get_advantage, get_prob
from params import PredictionData, Configs as conf
from tensorflow.keras.utils import to_categorical
from dataset import load_cifar100
import tensorflow as tf
from models import TYPE
import logging
import sys

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    model_type = sys.argv[1]

    train_data, train_labels, test_data, test_labels = load_cifar100()
    if model_type == 'alexnet':
        model = TYPE.ALEXNET(conf.input_shape, conf.num_classes)
    if model_type == 'densenet':
        model = TYPE.DENSENET(conf.input_shape, conf.num_classes)
    if model_type == 'vgg':
        model = TYPE.VGG(conf.input_shape, conf.num_classes)
    if model_type == 'resnet':
        model = TYPE.RESNET(conf.input_shape, conf.num_classes)

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=conf.epochs
    )

    # model.save('weights/c100_resnet')
    # model = keras.models.load_model('weights/')
    logging.info('Predict on train...')
    logits_train = model.predict(train_data)
    logging.info('Predict on test...')
    logits_test = model.predict(test_data)
    logging.info('Apply softmax to get probabilities from logits...')
    prob_train = tf.nn.softmax(logits_train, axis=-1)
    prob_test = tf.nn.softmax(logits_test)

    logging.info('Compute losses...')
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    y_train_onehot = to_categorical(train_labels)
    y_test_onehot = to_categorical(test_labels)
    loss_train = cce(
        constant(y_train_onehot), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(
        constant(y_test_onehot), constant(prob_test), from_logits=False).numpy()

    prediction_data = PredictionData(
        logits_test=logits_test,
        logits_train=logits_train,
        loss_test=loss_test,
        loss_train=loss_train,
        train_labels=train_labels,
        test_labels=test_labels
    )

    attacks_result, input, spec = get_report(prediction_data)
    logging.info(attacks_result.summary(by_slices=True))

    max_auc_attacker, max_advantage = get_advantage(attacks_result)
    membership_probability_results = get_prob(input, spec)

    logging.info("Attack type with max AUC: %s, AUC of %.2f, Attacker advantage of %.2f" %
                 (max_auc_attacker.attack_type,
                  max_auc_attacker.roc_curve.get_auc(),
                  max_auc_attacker.roc_curve.get_attacker_advantage()))

    # logging.info(membership_probability_results.summary(threshold_list=conf.threshold))
