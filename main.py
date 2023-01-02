# System
import argparse
import torch
import os
import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
# Custom
import util
from data_loader import Chest_Dataset, get_numpy_data2d_with_labels, get_numpy_data2d
import models

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root-dir", type=str, default='chest_xray/',
                    help="root path of data set")
parser.add_argument("--cnn-use-model", "-cum", type=str, default='resnet152',
                    help="use which model to train")
parser.add_argument("--load-model", "-lm", action='store_true', default=False,
                    help="load model")
parser.add_argument("--save-model-dir", "-smp", type=str, default='saves/',
                    help="save model directory path")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resize-size', '-rs', type=int, default=256, metavar='S',
                    help='resize size (default: (256, 256))')
parser.add_argument('--epochs', '-ep', default=200, type=int, metavar='N',
                    help='number of total initial epochs to run (default: 200)')
parser.add_argument('--start-epoch', '-sep', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=128, type=int,metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '-lr', default=0.1, type=float,
                    metavar='TLR', help='train learning rate')
parser.add_argument('--lr-drop-interval', '-lr-drop', default=50, type=int,
                    metavar='LRD', help='learning rate drop interval (default: 50)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
args = parser.parse_args()

# get data_loader
train_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((args.resize_size, args.resize_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: np.repeat(x, 3, axis=0)),  # copy 1 channel to generate 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((args.resize_size, args.resize_size)),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: np.repeat(x, 3, axis=0)),  # copy 1 channel to generate 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = Chest_Dataset(root_dir = args.root_dir, train_val_test = "train/", transform=train_transform)
val_set = Chest_Dataset(root_dir = args.root_dir, train_val_test = "val/", transform=test_transform)
test_set = Chest_Dataset(root_dir = args.root_dir, train_val_test = "test/", transform=test_transform)
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
train_data2d, train_labels = get_numpy_data2d_with_labels(train_loader)
val_data2d, val_labels = get_numpy_data2d_with_labels(val_loader)
test_data2d, img_idxs = get_numpy_data2d(test_loader)

# select device
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if use_cuda else 'cpu')
args.save_model_path = args.save_model_dir + f'{args.cnn_use_model}_checkpoint.tar'
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')


def svm_process():
    util.topic_log('SVM')
    model = svm.SVC()
    model.fit(train_data2d, train_labels)
    predicts = model.predict(val_data2d)
    util.evaluate_log(predicts, val_labels)


def kmeans_process():
    util.topic_log('Kmeans')
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(train_data2d)
    predicts = model.predict(val_data2d)
    util.evaluate_log(predicts, val_labels)


def random_foreset_process():
    util.topic_log('Random Forest')
    model = RandomForestClassifier()
    model.fit(train_data2d, train_labels)
    predicts = model.predict(val_data2d)
    util.evaluate_log(predicts, val_labels)

def cnn_process():
    use_model = args.cnn_use_model
    util.topic_log(use_model)
    if use_model == 'resnet50':
        pretrain_model = torch_models.resnet50(pretrained=True)
    elif use_model == 'resnet101':
        pretrain_model = torch_models.resnet101(pretrained=True)
    elif use_model == 'resnet152':
        pretrain_model = torch_models.resnet152(pretrained=True)
    elif use_model == 'alexnet':
        pretrain_model = torch_models.alexnet(pretrained=True)
    else:
        raise Exception
    model = models.FineTuneModel(pretrain_model, 'resnet' if 'resnet' in use_model else use_model).to(args.device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0001)
    best_top1 = 0
    if args.load_model:
        best_top1 = util.load_model(model, optimizer, args, args.save_model_path)
        print(f'load_model: {args.load_model} ({best_top1})')
    
    util.model_fit(model, optimizer, args, train_loader, val_loader, best_top1)
    _, predicts = util.val_epoch(model, args, val_loader, get_predicts=True)
    util.evaluate_log(predicts, val_labels)

def testing():
    use_model = args.cnn_use_model
    if use_model == 'resnet50':
        pretrain_model = torch_models.resnet50(pretrained=True)
    elif use_model == 'resnet101':
        pretrain_model = torch_models.resnet101(pretrained=True)
    elif use_model == 'resnet152':
        pretrain_model = torch_models.resnet152(pretrained=True)
    elif use_model == 'alexnet':
        pretrain_model = torch_models.alexnet(pretrained=True)
    else:
        raise Exception
    model = models.FineTuneModel(pretrain_model, 'resnet' if 'resnet' in use_model else use_model).to(args.device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0001)
    _ = util.load_model(model, optimizer, args, args.save_model_path)
    predicts = list()
    idxs = list()
    with torch.no_grad():
        for data, img_idx in test_loader:
            data = data.to(args.device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predicts += list(pred.cpu().numpy().reshape(-1))
            idxs.extend(img_idx)
        idxs = np.array(idxs)
    util.preds_to_csv(idxs, predicts)
    


def check_dir_exist():
    os.makedirs(f'{args.save_model_dir}', exist_ok=True)


def part1():
    util.topic_log('part1', 80)
    first_image_data, _ = train_set.images_with_labels[0]
    PIL_image = Image.fromarray(first_image_data)
    trans_image_data = train_set.transform(PIL_image)# / np.max(first_image_data))
    print(f'original image: size ({first_image_data.shape}), min ({np.min(first_image_data)}), max ({np.max(first_image_data)}), mean ({np.mean(first_image_data)})')
    print(f'transform image: size ({trans_image_data.shape}), min ({torch.min(trans_image_data)}), max ({torch.max(trans_image_data)}), mean ({torch.mean(trans_image_data)})')


def part2():
    util.topic_log('part2', 80)
    svm_process()
    kmeans_process()
    random_foreset_process()
    cnn_process()
    


def main():
    #check_dir_exist()
    #part1()
    #part2()
    testing()

if __name__ == '__main__':
    main()
