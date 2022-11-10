import matplotlib.pyplot as plt
import numpy as np
import seaborn  as sns

def plot_loss(loss, mode):
    plt.plot(loss['train'])
    plt.plot(loss['val'])
    plt.title('model train and val Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val'], loc='upper left')
    plt.savefig(f'{mode}_loss.png')

def plot_cifar_probs(test, train, in_train_acc, non_train_acc):        
    #max_probs = np.amax(probs, axis=1)   
    test_preds = np.argmax(test, axis=1)
    train_preds = np.argmax(train, axis=1)
    non_members_counts = [np.count_nonzero(test_preds == i) for i in range(100)]
    members_counts = [np.count_nonzero(train_preds == i) for i in range(100)]
    max_test_probs = np.amax(test, axis=1)
    max_train_probs = np.amax(train, axis=1)
    print('after softmax', max_test_probs)
    # fig, ax = plt.subplots()
    
    # sns.histplot(max_test_probs)
    # ax.set_xlim([0, 100])   
    # ax.set(xlabel='Normal Distribution', ylabel='Frequency')   
    
    
    plt.figure(figsize=(10, 8))    
    plt.bar([i for i in range(len(non_members_counts))], non_members_counts, color='grey')
    plt.bar([i for i in range(len(members_counts))], members_counts, color='olive')
    plt.xlabel("Classes")
    plt.ylabel("Occurance")
    #plt.legend(["PDF on non-member data ({:.2f}%)".format(non_train_acc), "PDF on member data ({:.2f}%)".format(in_train_acc)])
    #plt.savefig('target_probs_SOFTMAX.png')
    
def plot_pred_diff(x, y):
            
    # for i in range(2):            
    #     plt.scatter(x, y, label=str(i))
    # plt.legend()
    # plt.savefig('./plots/preds.png')
    
    plt.plot(x)    
    plt.plot(y)
    plt.ylabel('Classes')
    plt.xlabel('Data inputs')
    plt.legend(['negative distance', 'positive distance'],  loc='upper left')
    plt.title('positive and negative samples')
    plt.savefig('plots/preds.png')