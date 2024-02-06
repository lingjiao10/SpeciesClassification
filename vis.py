from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

from datasets.inaturalist import INaturalist
import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as T

import matplotlib.pyplot as plt
import random
import collections


def get_dict(path):
	checkpoint = torch.load(path)
	# model.load_state_dict(checkpoint['state_dict'])

	new_state_dict = collections.OrderedDict()
	for k, v in checkpoint['state_dict'].items():
	    name = k.replace('module.', '')# remove `module.`
	    new_state_dict[name] = v
	return new_state_dict


def get_vis(model, input_tensor, target, rgb_img_01):
	target_layers = [model.layer4[-1]]

	outputs = model(input_tensor.unsqueeze(0))
	_, preds = torch.max(outputs, 1) #preds为最大值的索引
	

	# Create an input tensor image for your model..
	# Note: input_tensor can be a batch tensor with several images!

	# Construct the CAM object once, and then re-use it on many images:
	# args.use_cuda = False
	cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

	# You can also use it within a with statement, to make sure it is freed,
	# In case you need to re-create it inside an outer loop:
	# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
	#   ...

	# We have to specify the target we want to generate
	# the Class Activation Maps for.
	# If targets is None, the highest scoring category
	# will be used for every image in the batch.
	# Here we use ClassifierOutputTarget, but you can define your own custom targets
	# That are, for example, combinations of categories, or specific outputs in a non standard model.

	# targets = [ClassifierOutputTarget(273)]
	targets = [ClassifierOutputTarget(target)]

	# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
	grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)

	# In this example grayscale_cam has only one image in the batch:
	grayscale_cam = grayscale_cam[0, :]
	
	visualization = show_cam_on_image(rgb_img_01, grayscale_cam, use_rgb=True)

	return visualization, preds

def get_parents(tree, node_id, parent_names = []):
	parent = tree.parent(node_id)
	if not parent.is_root():
		parent_names.append(parent.tag)
		get_parents(tree, parent.identifier, parent_names)

def get_ancestors(tree, node_id):
	parent_names = []
	get_parents(tree, node_id, parent_names)
	return parent_names



def randomVIS_Order_CE_vs_treelayerloss():
	target_type = ["order", "class", "phylum", "kingdom"]
	nature = INaturalist("h:/Datasets/iNaturalist/", version='2021_valid',target_type=target_type, build_tree=True, load_weight=False)
	# model = resnet50(pretrained=True)
	model = resnet50(num_classes=len(nature.categories_index[target_type[0]]))
	model.load_state_dict(get_dict('./runs/train/exp-20230302-1/checkpoint.pth.tar'))
	model_h = resnet50(num_classes=len(nature.categories_index[target_type[0]]))
	model_h.load_state_dict(get_dict('./runs/train/exp-202300711-1/model_best.pth.tar'))


	for i in range(10):
		idx = random.randint(0, len(nature)-1)
		# print(idx)
		rgb_img, target = nature[idx]
		input_tensor = T.pil_to_tensor(rgb_img).to(torch.float32) / 255
		rgb_img_01 = input_tensor.numpy().transpose((1, 2, 0))

		target_name = []
		for t in range(len(target_type)):
			target_name.append(list(nature.categories_index[target_type[t]].keys())[target[t]])


		vis1, preds1 = get_vis(model, input_tensor, target[0], rgb_img_01)
		vis2, preds2 = get_vis(model_h, input_tensor, target[0], rgb_img_01)

		preds1_name = list(nature.categories_index['order'].keys())[preds1]
		preds2_name = list(nature.categories_index['order'].keys())[preds2]

		pred1_full_name = [preds1_name] + get_ancestors(nature.category_tree, "{}_{}".format(preds1.item(), preds1_name))
		pred2_full_name = [preds2_name] + get_ancestors(nature.category_tree, "{}_{}".format(preds2.item(), preds2_name))


		plt.figure("Image")
		plt.subplot(1,3,1)
		plt.title(target_name)
		plt.imshow(rgb_img_01)
		plt.subplot(1,3,2)
		plt.title(pred1_full_name)
		plt.imshow(vis1)
		plt.subplot(1,3,3)
		plt.title(pred2_full_name)
		plt.imshow(vis2)
		plt.show()

def randomVIS_speices_CE():
	target_type = ["full"]
	nature = INaturalist("h:/Datasets/iNaturalist/", version='2021_valid',target_type=target_type, 
		build_tree=True, tree_depth=7, load_weight=False)
	# model = resnet50(pretrained=True)
	model = resnet50(num_classes=len(nature.all_categories))
	model.load_state_dict(get_dict('./runs/train/exp-20230822-1/model_best.pth.tar'))
	# model_h = resnet50(num_classes=len(nature.categories_index[target_type[0]]))
	# model_h.load_state_dict(get_dict('./runs/train/exp-202300711-1/model_best.pth.tar'))


	for i in range(10):
		idx = random.randint(0, len(nature)-1)
		# print(idx)
		rgb_img, target = nature[idx]
		input_tensor = T.pil_to_tensor(rgb_img).to(torch.float32) / 255
		rgb_img_01 = input_tensor.numpy().transpose((1, 2, 0))

		target_name = nature.all_categories[target]
		# for t in range(len(target_type)):
		# 	target_name.append(list(nature.categories_index[target_type[t]].keys())[target[t]])
		# 	# target_name.append(list(nature.categories_index[target_type[t]].keys())[target[t]])


		vis1, preds1 = get_vis(model, input_tensor, target, rgb_img_01)
		# vis2, preds2 = get_vis(model_h, input_tensor, target[0], rgb_img_01)

		preds1_name = nature.all_categories[preds1]
		# preds2_name = list(nature.categories_index[target[0]].keys())[preds2]
		pred1_full_name = preds1_name

		# pred1_full_name = [preds1_name] + get_ancestors(nature.category_tree, "{}_{}".format(preds1.item(), preds1_name))
		# pred2_full_name = [preds2_name] + get_ancestors(nature.category_tree, "{}_{}".format(preds2.item(), preds2_name))


		plt.figure("Image")
		plt.subplot(1,2,1)
		plt.title(target_name)
		plt.imshow(rgb_img_01)
		plt.subplot(1,2,2)
		plt.title(pred1_full_name)
		plt.imshow(vis1)
		# plt.subplot(1,3,3)
		# plt.title(pred2_full_name)
		# plt.imshow(vis2)
		plt.show()


if __name__ == '__main__':
    randomVIS_speices_CE()