from models.attack import RandomForest, DistanceAttack
from models.shadow import ShadowModel
from models.target import DenseNet
from dataset import *
from actions import *
import dataset
import conf
import sys

if __name__ == '__main__':    
    arg = sys.argv[1]
    dataloader = dataset.Cifar100Dataset()
    dataloader = dataloader.get_dataset()
    target = DenseNet(conf.cifar_classes) 
    #target_path = target.train(dataloader)
    target_path = './weights/target_cifar100_min.pt'
    shadow = ShadowModel(conf.cifar_classes)
    shadow_path = './weights/shadow_cifar100_min.pt'
    #shadow_path = shadow.train(dataloader)    
    
    sh_train_labels, sh_train_logits = shadow.test(shadow_path, dataloader.shadow_trainloader)
    sh_test_labels, sh_test_logits = shadow.test(shadow_path, dataloader.shadow_valloader)
   
    t_train_labels, t_train_logits = target.test(target_path, dataloader.target_trainloader)
    t_test_labels, t_test_logits = target.test(target_path, dataloader.target_valloader)

    if arg == 'rf':
        attack_train_data = prepare_attack_data(sh_train_logits, sh_test_logits)
        attack_test_data = prepare_attack_data(t_train_logits, t_test_logits)
        attacker = RandomForest()
        #attacker.train(attack_train_data)
        attacker.test(attack_test_data)
    else:
        pos_train, pos_train_, neg_train = prepare_nn_attack_data(sh_train_logits, sh_test_logits)
    
        attacker = DistanceAttack(n_classes=conf.cifar_classes)  
        train_attacker(attacker, pos_train, pos_train_, neg_train)
        #test_attacker(attacker, attack_test_data)
    

    
    
        
   
    
    

