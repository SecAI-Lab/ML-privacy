from models.attacker import EnsembleAttacker
from models.shadow import ShadowModel
from models.target import DenseNet
from dataset import *
from actions import *
from dataset import get_CifarDataloader
from conf import CifarConf


if __name__ == '__main__':
    dataloader = get_CifarDataloader(c=CifarConf)
    target = DenseNet(CifarConf.n_classes)
    shadow = ShadowModel(CifarConf.n_classes)

    # target.train(dataloader, CifarConf.target_path)
    # shadow.train(dataloader, CifarConf.shadow_path)

    sh_train_logits, sh_train_tru_preds = shadow.test(
        CifarConf.shadow_path, dataloader.shadow_trainloader)
    sh_test_logits, sh_test_tru_preds = shadow.test(
        CifarConf.shadow_path, dataloader.shadow_valloader)

    t_train_logits, t_train_tru_preds = target.test(
        CifarConf.target_path, dataloader.target_trainloader)
    t_test_logits, t_test_tru_preds = target.test(
        CifarConf.target_path, dataloader.target_valloader)
    attacker_path = CifarConf.attacker_path

    attack_train_data = prepare_attack_data(sh_train_logits, sh_test_logits)
    attack_test_data = prepare_attack_data(t_train_logits, t_test_logits)

    attacker = EnsembleAttacker(attacker_path)
    attacker.train(attack_train_data)
    attacker.test(attack_test_data)
