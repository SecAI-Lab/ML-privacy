from report import get_report, get_advantage, get_prob
from params import PredictionData, Configs as conf
from tensorflow.keras.utils import to_categorical
from dataset import load_cifar100, load_texas100, load_reuters
import tensorflow as tf
from models import TYPE
import logging
import sys
from utils import plot_loss

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    model_type = sys.argv[1]

    #train_data, train_labels, test_data, test_labels = load_cifar100()
    t_train_data, t_train_labels, t_test_data, t_test_labels, sh_train_data, sh_train_labels, sh_test_data, sh_test_labels = load_cifar100()
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if model_type == 'densenet':
        target_model = TYPE.DENSENET(conf.input_shape, conf.num_classes)
        shadow_model = TYPE.DENSENET(conf.input_shape, conf.num_classes)
    if model_type == 'texas':
        model = TYPE.TEXAS()
        test_data, test_labels, _, _, train_data, train_labels = load_texas100()
    if model_type == 'reuter':
        model = TYPE.REUTER()
        train_data, train_labels, test_data, test_labels = load_reuters()

    target_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    target_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=conf.target_path,
                                                            save_weights_only=True,
                                                            verbose=1)
    shadow_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=conf.shadow_path,
                                                            save_weights_only=True,
                                                            verbose=1)

    logging.info('Training target model')
    target_model.fit(
        t_train_data,
        validation_data=t_test_data,
        epochs=conf.epochs,
        batch_size=conf.batch_size,
        callbacks=[target_cp_callback]
    )

    #plot_loss(history, model_type)
    logging.info('Training shadow model')

    shadow_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    shadow_model.fit(
        sh_train_data,
        epochs=conf.epochs,
        validation_data=sh_test_data,
        batch_size=conf.batch_size,
        callbacks=[shadow_cp_callback]
    )

    # if model_type == 'texas':
    #     logits_train = shadow.predict(train_attack_data)
    # else:
    logging.info(f'loading target_model {conf.target_path}')
    target_model.load_weights(conf.target_path)

    logging.info(f'loading shadow_model {conf.shadow_path}')
    shadow_model.load_weights(conf.shadow_path)

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
    print('Loss train', loss_train, loss_train.shape)
    print('Loss test', loss_test, loss_test.shape)
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

    # max_auc_attacker, max_advantage = get_advantage(attacks_result)
    # membership_probability_results = get_prob(input, spec)

    # logging.info("Attack type with max AUC: %s, AUC of %.2f, Attacker advantage of %.2f" %
    #              (max_auc_attacker.attack_type,
    #               max_auc_attacker.roc_curve.get_auc(),
    #               max_auc_attacker.roc_curve.get_attacker_advantage()))

    # logging.info(membership_probability_results.summary(threshold_list=conf.threshold))
