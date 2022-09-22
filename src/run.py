import dataset
import conf
from models.target import DenseNet



if __name__ == '__main__':
    trainloader, testloader = dataset.test_dataset()
    model = DenseNet(num_classes=conf.cifar_classes) 
    model.train(trainloader, testloader)
   
    
    

