from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torchvision.models as models

# from datasets.inaturalist import INaturalist
import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as T
import torch.cuda as cuda

import matplotlib.pyplot as plt
import random
import collections
import os

from datasets.vegetablepests import VegetablePests

 
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'


def get_dict(path):
    checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['state_dict'])

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k.replace('module.', '')# remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_vis(model, target_layers, input_tensor, target, rgb_img_01):

    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))
        _, preds = torch.max(outputs, 1) #preds为最大值的索引
    

    print("------------finish preds----------")
    print_gpu(0)
    torch.cuda.empty_cache()
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # Construct the CAM object once, and then re-use it on many images:
    # args.use_cuda = False


    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:

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

def print_gpu(i):
    props = cuda.get_device_properties(i)
    total_memory = props.total_memory // 1024**2  # 转换为MB
    free_memory = cuda.memory_allocated(i) // 1024**2  # 转换为MB
    memory_cached = cuda.memory_cached(i) // 1024**2 
    print("GPU {} 总内存: {}MB".format(i, total_memory))
    print("GPU {} memory_allocated: {}MB".format(i, free_memory))
    print("GPU {} memory_cached: {}MB".format(i, memory_cached))


def get_model(arch, ckpt, num_classes):
    model = models.__dict__[arch](num_classes=num_classes)
    model.load_state_dict(get_dict(ckpt))

    # model = resnet50(num_classes=len(dataset.classname_df))
    # model.load_state_dict(get_dict('./runs/train/exp-20240529-1/model_best.pth.tar'))
    # model.load_state_dict(get_dict('./runs/train/exp-20230822-1/model_best.pth.tar'))
    # model_h = resnet50(num_classes=len(nature.categories_index[target_type[0]]))
    # model_h.load_state_dict(get_dict('./runs/train/exp-202300711-1/model_best.pth.tar'))

    model.to('cuda:0')
    model.eval()
    print("------------finish loading models----------")
    print_gpu(0)

    if arch=="resnet50":
        target_layers = [model.layer4[-1]]
    elif arch == "vit_l_16":
        # target_layers = [model.blocks[-1].norm1]
        target_layers = [model.conv_proj]
    else:
        target_layers = None

    return model, target_layers

def randomVIS_speices_CE():
    print("------------start----------")
    print_gpu(0)
    # target_type = ["full"]
    # nature = INaturalist("h:/Datasets/iNaturalist/", version='2021_valid',target_type=target_type, 
    #   build_tree=True, tree_depth=7, load_weight=False)
    # model = resnet50(pretrained=True)
    root_dir = "/home/h3c/wangcong/.cache/lavis/vegetable_pests"
    # root_dir = "H:/Datasets/vegetable_pests"
    dataset = VegetablePests(root=root_dir+'/images/test', 
            class_name_txt=root_dir+"/classnames.txt")

    num_classes = len(dataset.classname_df)

    resnet50, target_layers_resnet = get_model(
        "resnet50", './runs/train/exp-20240529-1/model_best.pth.tar', num_classes
        )

    vit, target_layers_vit =  get_model(
        "vit_l_16", './runs/train/exp-20240529-3/model_best.pth.tar', num_classes
        )
    

    mean = np.float32([0.485, 0.456, 0.406])
    std = np.float32([0.229, 0.224, 0.225])
    
    for i in range(10):
        torch.cuda.empty_cache()
        idx = random.randint(0, len(dataset)-1)
        # print(idx)
        rgb_img, target = dataset[idx]
        rgb_img = rgb_img.resize((224, 224))
        input_tensor = T.pil_to_tensor(rgb_img).to(torch.float32) / 255
        
        rgb_img_01 = input_tensor.numpy().transpose((1, 2, 0))

        input_tensor = (input_tensor - mean.reshape(3,1,1)) / std.reshape(3,1,1)

        print('input_tensor.size() = ', input_tensor.size())

        target_name = dataset.classname_df.loc[target, 'scientific_name']
        print("target:", target_name)
        # for t in range(len(target_type)):
        #   target_name.append(list(nature.categories_index[target_type[t]].keys())[target[t]])
        #   # target_name.append(list(nature.categories_index[target_type[t]].keys())[target[t]])

        input_tensor = input_tensor.to('cuda:0')

        print("------------finish loading input_tensor----------")
        print_gpu(0)

        vis1, preds1 = get_vis(resnet50, target_layers_resnet, input_tensor, target, rgb_img_01)
        vis2, preds2 = get_vis(vit, target_layers_vit, input_tensor, target, rgb_img_01)

        print("------------finish calculating cam----------")
        print_gpu(0)

        # vis2, preds2 = get_vis(model_h, input_tensor, target[0], rgb_img_01)
        preds1_name = dataset.classname_df.loc[int(preds1), 'scientific_name']
        print("predict: ", preds1_name)
        # preds1_name = nature.all_categories[preds1]
        # preds2_name = list(nature.categories_index[target[0]].keys())[preds2]
        # pred1_full_name = preds1_name

        preds2_name = dataset.classname_df.loc[int(preds2), 'scientific_name']

        # pred1_full_name = [preds1_name] + get_ancestors(nature.category_tree, "{}_{}".format(preds1.item(), preds1_name))
        # pred2_full_name = [preds2_name] + get_ancestors(nature.category_tree, "{}_{}".format(preds2.item(), preds2_name))


        plt.figure("Image")
        plt.subplot(1,3,1)
        plt.title(target_name)
        plt.imshow(rgb_img_01)
        plt.subplot(1,3,2)
        plt.title(preds1_name)
        plt.imshow(vis1)
        plt.subplot(1,3,3)
        plt.title(preds2_name)
        plt.imshow(vis2)
        # plt.show()

        save_path = "./vis"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, "veg_{}.png".format(i)))
        plt.close()
        
        print_gpu(0)


if __name__ == '__main__':
    # model_arch = "resnet50"
    # ckpt = './runs/train/exp-20240529-1/model_best.pth.tar'

    # model_arch = "vit_l_16"
    # ckpt = './runs/train/exp-20240529-3/model_best.pth.tar'



    randomVIS_speices_CE()
    print("------------finish all.----------")
    print_gpu(0)