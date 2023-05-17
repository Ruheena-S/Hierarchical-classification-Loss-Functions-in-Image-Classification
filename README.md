# Hierarchical classification Loss Functions in Image-Classification

Numerous multi-class classification tasks are carried out in the real world, but most of these algorithms use 0-1 loss or its surrogates, which ignore the hierarchy involved in the data. These performance metrics ignore the degree to which the predicted class label and the true class label are similar and evaluate the classes other than the truth value as incorrect. So a taxonomic hierarchical tree with a well defined hierarchy of class labels is provided to include the hierarchy in the evaluation so that the model may learn the unique traits of the classes that are subclasses of the same superclass.

### Multi-class loss functions

Multi-class classification is predictive modelling problem where inputs are predicted as one of n (>2) classes. The problem is often implemented as predicting the probability of the input belonging to each known class. Loss is a metric measuring a model's effectiveness. The model strives to achieve the lowest loss while learning.

Here apart from flat classification - **cross entropy, One vs All (OvA)** we learn different loss functions like **OvA cascade, Binary Encoded Prediction (BEP) cascade** losses which considers the hierarachy of data in evaluation.

### Experiments

In our experiments we have used three model architechtures for feature extraction.
- VGG 16
- ResNet 18
- ResNet 50

### Data Preparation

Datasets – CIFAR100, Mammal ImageNet, Tiered ImageNet, iNaturalist’19 and ImageNet
CIFAR100 – pytorch has this dataset. Torchvision -> datasets->CIFAR100.
Mammal ImageNet – Download the dataset: https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html
Tiered ImageNet – Download the dataset: https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html
iNaturalist’19 – Download the dataset:  https://www.kaggle.com/competitions/inaturalist-2019-fgvc6/data
ImageNet – https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be

### Requirements

pip3 install torch torchvision 
pip3 install torch_optimizer torchmetrics
Pip install nvidia
sudo apt-get -y install python3-nltk
pip install conditional
sudo apt-get -y install libjs-json
pip install tensorboardX

### How to Run the code

Python3 <filename.py>
Make sure you change the data paths in the code file accordingly. 

### Observation
The reliable substitutes for tree distance loss, such as OvA cascade and BEP cascade,  outperform flat classification algorithms that ignore hierarchy.

### References
- Harish G. Ramaswamy, Ambuj Tewari and Shivani Agarwal, ”Consistent Algorithms for Multiclass classification with an Abstain option”  https://www.shivaniagarwal.net/Publications/2018/ejs-18-multiclass-abstain.pdf
- Harish G. Ramaswamy, Ambuj Tewari and Shivani Agarwal, ”Convex calibrated surrogates for Hierarchical classification” https://ambujtewari.github.io/research/ramaswamy15convex.pdf
