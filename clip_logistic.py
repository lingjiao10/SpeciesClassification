import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from datasets.inaturalist import INaturalist



def load_inat_dataset():
    # Load the dataset
    # root = os.path.expanduser("~/.cache")
    # train = CIFAR100(root, download=True, train=True, transform=preprocess)
    # test = CIFAR100(root, download=True, train=False, transform=preprocess)
    # root_dir = 'H:/Datasets/iNaturalist'
    root_dir = '/home/Datasets/iNaturalist'
    # target_type = ["kingdom", "phylum", "class", "order", "family", "genus", "full"]
    target_type = "full"
    train_dataset = INaturalist(root=root_dir, version='2021_train_mini', target_type=target_type, transform=preprocess)
    val_dataset = INaturalist(root=root_dir, version='2021_valid', target_type=target_type, transform=preprocess)

def load_vege_dataset():
    

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=1000)):
            features = model.encode_image(images.to(device))

            # print("labels, ", labels)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train_dataset)
test_features, test_labels = get_features(val_dataset)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
LR = classifier.fit(train_features, train_labels)
print("LR.n_iter_: ", LR.n_iter_)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clip LogisticRegression code.')
    parser.add_argument('--dataset', default='iNaturalist', help='dataset type')

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    model, preprocess = clip.load('ViT-B/32', device)
