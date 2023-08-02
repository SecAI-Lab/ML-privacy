## ML-privacy

Privacy risk assesment tool on Deep learning models 

### How to use

        cd src/

        python attack.py

<b>Available attack</b>: MIA (Memebership Inference)

<b>Default attacker</b>: EnsembleAttacker 

<b>Target model</b>: DenseNet121

<b>Dataset</b>: Cifar10 (default) and Cifar100

To enable GPU:

        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html