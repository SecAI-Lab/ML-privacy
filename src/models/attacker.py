from sklearn import ensemble
import joblib
from utils import utils
import os
import logging

logging.basicConfig(level=logging.INFO)


class EnsembleAttacker:
    """Classical ML based MIA model"""

    def __init__(self, attackers_path) -> None:
        self.rf = ensemble.RandomForestClassifier(n_jobs=1)
        self.rf.set_params(
            n_estimators=200, criterion='gini', max_features='sqrt')
        self.path = attackers_path

    def train(self, train_data):
        x, y = utils.split_data(train_data)
        logging.info(f'Training attacker Random Forest...')
        self.rf.fit(x, y)
        joblib.dump(self.rf, self.path)
        logging.info(f'Saving attacker to {self.path}')

    def test(self, test_data):
        x, y = utils.split_data(test_data)
        logging.info(f'Testing attack model')
        if os.path.exists(self.path):
            model = joblib.load(self.path)
        else:
            model = model
        preds = model.predict(x)
        tru_preds = 0
        for t, p in zip(y, preds):
            tru_preds += 1 if (p == t) else 0
        print("Acc of attacker: {}, {}/{}".format(tru_preds /
                                                  len(preds), tru_preds, len(preds)))
