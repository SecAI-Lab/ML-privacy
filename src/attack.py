from models.attacker import EnsembleAttacker, NNAttack
from models.shadow import ShadowModel
from models.target import DenseNet, GRU
from dataset import *
from actions import *
import dataset
import conf
import sys

if __name__ == '__main__':
    arg = sys.argv[1]

    if arg == 'cnn':
        cifar = dataset.Cifar100Dataset()
        dataloader = cifar.get_dataset()
        target = DenseNet(conf.cifar_classes)
        shadow = ShadowModel(conf.cifar_classes)

        # target.train(dataloader)
        # shadow.train(dataloader)

        sh_train_logits, sh_train_tru_preds = shadow.test(
            conf.cifar_shadow_path, dataloader.shadow_trainloader)
        sh_test_logits, sh_test_tru_preds = shadow.test(
            conf.cifar_shadow_path, dataloader.shadow_valloader)

        t_train_logits, t_train_tru_preds = target.test(
            conf.cifar_target_path, dataloader.target_trainloader)
        t_test_logits, t_test_tru_preds = target.test(
            conf.cifar_target_path, dataloader.target_valloader)
        attacker_path = conf.attackers_path

    if arg == 'rnn':
        imdb = dataset.ImdbReviewsDataset()
        dataloader = imdb.get_dataset()
        model = GRU(conf.n_layers, conf.hidden_dim, imdb.vocab_size,
                    conf.emdeb_dim, imdb.n_clases).to(conf.device)
        #train_model(dataloader, model, path=conf.imdb_shadow_path, mode='shadow')

        t_train_logits = test_model(
            dataloader.target_trainloader, model, PATH=conf.imdb_target_path)
        t_test_logits = test_model(
            dataloader.target_valloader, model, PATH=conf.imdb_target_path)

        sh_train_logits = test_model(
            dataloader.shadow_trainloader, model, PATH=conf.imdb_shadow_path)
        sh_test_logits = test_model(
            dataloader.shadow_valloader, model, PATH=conf.imdb_shadow_path)
        attacker_path = conf.attack_for_rnn

    attack_train_data = prepare_attack_data(sh_train_logits, sh_test_logits)
    attack_test_data = prepare_attack_data(t_train_logits, t_test_logits)

    attacker = EnsembleAttacker(attacker_path)
    attacker.train(attack_train_data)
    attacker.test(attack_test_data)