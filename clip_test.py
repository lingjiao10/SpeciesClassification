import os
import argparse
import clip
import torch
# from torchvision.datasets import CIFAR100

import numpy as np
import pandas as pd
from datasets.inaturalist import INaturalist
import random
from tqdm import tqdm

from datasets.vegetablepests import VegetablePests

# Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def get_dataset(ds_name):
    if ds_name == "iNaturalist":
        gt_idx = 6
        # root_dir = 'H:/Datasets/iNaturalist'
        root_dir = '/home/Datasets/iNaturalist'
        val_dataset = INaturalist(root=root_dir, version='2021_valid', target_type=["kingdom", "phylum", "class", "order", "family", "genus", "full"])
    elif ds_name == "mydata":
        gt_idx = 0
        val_dataset =  VegetablePests(root="~/wangcong/Datasets/vegetable_pests")

    return val_dataset, gt_idx

def predict(val_dataset, id, text_features, gt_idx=6):
    """
    gt_idx: index of ground truth in target // inat=6, vege=0
    """

    # Prepare the inputs
    # image, class_id = cifar100[3637]
    # id = random.randint(0, len(val_dataset)-1)
    image, class_id = val_dataset[id]

    image_input = preprocess(image).unsqueeze(0).to(device)
    


    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # print("image_features.shape: ", image_features.shape)
    # print("text_features.shape: ", text_features.shape)




    # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{name_list[index]:>80s}: {100 * value.item():.2f}%")

    gt_id = class_id[gt_idx]
 
    top1 = 0
    top5 = 0
    if gt_id in indices:
        top5 = 1
        if gt_id == indices[0]:
            top1 = 1

    return top1, top5

# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
   
""" 
calculate text features for inaturalist dataset(1w classes)
"""
def cal_text_features(dataset, classname_df=None):
    if dataset == "mydata":
        with torch.no_grad():
            name_list = []
            # prompt_template = "a photo of {}, whose scientific name is {}."
            prompt_template = "a photo of {}."
            for i in range(classname_df.shape[0]):
                common_name = classname_df.at[i, 'common_name1']
                if not common_name: common_name = "insect"
                # name_list.append(prompt_template.format(common_name, classname_df.at[i, 'scientific_name']))
                name_list.append(prompt_template.format(common_name))
     
            text_inputs = clip.tokenize(name_list).to(device)
            text_features = model.encode_text(text_inputs)
    else:
        with torch.no_grad():
            if not os.path.exists("inat_text_features.pt"):
                #read catogry table
                df = pd.read_csv("./cat_table.csv", index_col=0)
                name_list = []
                # prompt_template = "a {} {} from {}, {}, {}, {}, {}"
                # prompt_template = "a photo of {} {} from kingdom of {}, phylum of {}, class of {}, order of {}, family of {}"
                # prompt_template = "a photo of Spodoptera litura, which is in the family Noctuidae, order Lepidoptera, class Insecta, phylum Arthropoda, kingdom Animalia."
                prompt_template = "a photo of {} {}, which is a speicies in the family {}, order {}, class {}, phylum {}, kingdom {}."
                
                for i in range(df.shape[0]):
                # for i in range(100):
                    # name_list.append(prompt_template.format(df.at[i, 'genus'], df.at[i, 'speicies'], 
                    #     df.at[i, 'kingdom'], df.at[i, 'phylum'], df.at[i, 'class'], df.at[i, 'order'], 
                    #     df.at[i, 'family']))
                    name_list.append(prompt_template.format(df.at[i, 'genus'], df.at[i, 'speicies'],                 
                        df.at[i, 'family'],  df.at[i, 'order'], df.at[i, 'class'], df.at[i, 'phylum'], df.at[i, 'kingdom']))
         
                text_inputs = clip.tokenize(name_list).to(device)
                text_features = model.encode_text(text_inputs)
                #save text_features
                torch.save(text_features, "inat_text_features.pt")
            else:
                print("loading inat_text_features.pt")
                text_features = torch.load("inat_text_features.pt")

    return text_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clip testing code.')
    parser.add_argument('--dataset', default='iNaturalist', help='dataset type')
    # Download the dataset
    # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    
    # args = parse_arguments()
    args = parser.parse_args()


    val_dataset, gt_idx = get_dataset(args.dataset)
    result_top1 = 0
    result_top5 = 0
    dataset_length = len(val_dataset)
    for id in tqdm(range(dataset_length)):
        # print(val_dataset.classname_df)
        top1, top5 = predict(val_dataset, id, cal_text_features(args.dataset, val_dataset.classname_df), gt_idx)
        result_top1 += top1
        result_top5 += top5

    acc1 = result_top1/len(val_dataset)
    acc5 = result_top5/len(val_dataset)
    print("top1 accuracy: {:.2f}%".format(acc1*100), ", top5 accuracy: {:.2f}%".format(acc5*100))



"""
python clip_test.py --dataset iNaturalist

python clip_test.py --dataset mydata
"""
