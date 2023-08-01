import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_loss(loss, label):
    plt.plot(loss['train'])
    plt.plot(loss['val'])
    plt.title('model train and val Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val'], loc='upper left')
    plt.savefig(f'{label}_loss.png')


def plot_cifar_probs(test, train, in_train_acc, non_train_acc):
    #max_probs = np.amax(probs, axis=1)
    test_preds = np.argmax(test, axis=1)
    train_preds = np.argmax(train, axis=1)
    non_members_counts = [np.count_nonzero(
        test_preds == i) for i in range(100)]
    members_counts = [np.count_nonzero(train_preds == i) for i in range(100)]
    max_test_probs = np.amax(test, axis=1)
    max_train_probs = np.amax(train, axis=1)
    print('after softmax', max_test_probs)
    # fig, ax = plt.subplots()

    # sns.histplot(max_test_probs)
    # ax.set_xlim([0, 100])
    # ax.set(xlabel='Normal Distribution', ylabel='Frequency')

    plt.figure(figsize=(10, 8))
    plt.bar([i for i in range(len(non_members_counts))],
            non_members_counts, color='grey')
    plt.bar([i for i in range(len(members_counts))],
            members_counts, color='olive')
    plt.xlabel("Classes")
    plt.ylabel("Occurance")
    # plt.legend(["PDF on non-member data ({:.2f}%)".format(non_train_acc), "PDF on member data ({:.2f}%)".format(in_train_acc)])
    # plt.savefig('target_probs_SOFTMAX.png')


def plot_pred_diff(x, y):    
    #sns.boxplot(data=[x, y])
    # sns.violinplot(data=[x, y])
    # plt.xticks([0, 1], ['Dataset 1', 'Dataset 2'])
    # plt.xlabel('Dataset')
    # plt.ylabel('Logits')        
    # sns.boxplot(data=[x, y])
    plt.hist(x, bins=50, label='Train/Seen Data (Acc: 0.93)', color = 'lightblue')
    plt.hist(y, bins=50, label='Test/Unseen Data (Acc: 0.65)', color = 'green')
    #plt.xticks([0, 1], ['Train/Seen Data (Acc: 0.93)', 'Test/Unseen Data (Acc: 0.65)'])   
    plt.xlabel('Logits')
    plt.ylabel('Count')
    plt.title('Shadow Model preformance for seen and unseen data')
    plt.legend()
    plt.show()
    
    #plt.savefig('shadow.png')

def plot_attackers_result():
    import seaborn as sns
    tf_privacy = 0.64
    priv_meter = 0.61
    ml_doctor = 0.74
    mi_attack = 0.62
    ours = 0.71
    attacks = ['TF Privacy', 'Privacy Meter', 'ML-Doctor', 'MI-attack', 'Distance based (ours)']
    scores = [tf_privacy, priv_meter, ml_doctor, mi_attack, ours]
    
    fig, ax = plt.subplots()
    y_pos = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = sns.color_palette('hls',len(attacks))
    
    hbars = ax.barh(y_pos, scores, align='center', color=colors, label=attacks, height=0.3)
    # ax.set_yticks(y_pos, labels=attacks)
    #ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Performance')
    ax.set_title('Existing MIA Attack scores on Cifar100 dataset')    
    ax.bar_label(hbars, fmt='%.2f')
    ax.set_xlim(right=1)
    ax.set_ylim(top=3.3)
    plt.legend()
    plt.savefig('attacks.png')

#plot_attackers_result()