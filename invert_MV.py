import os
import logging
import argparse
import pickle
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import model

import optuna

# define the data_loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

def deprocess_image(x):
    _x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # normalize
    _x -= _x.mean()
    _x /= (_x.std() + 1e-5)
    _x *= 0.25

    # clip to [0, 1]
    _x += 0.5
    _x = np.clip(_x, 0, 1)
    return _x

def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

def lp_norm(x, alpha):
    """ calculate LP-norm loss """
    return torch.abs(x.view(-1) ** alpha).sum() / np.prod(x.shape)

def tv_norm(x, beta):
    """ calculate total variation loss """
    a = (x[:, :, 1:, :-1] - x[:, :, :-1, :-1]) ** 2
    b = (x[:, :, :-1, 1:] - x[:, :, :-1, :-1]) ** 2
    return torch.sum((a + b) ** (beta / 2.0)) / np.prod(x.shape)

def invert(model2, y, params, device):
    # initial image
    x = torch.rand(1, 3, 32, 32).to(device).requires_grad_(False)

    # variable change [Carlini & Wagner, 2017]
    if opt.variable_change:
        w = 0.5 * torch.log(x / (1 - x))
    else:
        w = x.clone()
    w.requires_grad_(requires_grad=True)

    # set loss and optimizer
    loss_f = nn.MSELoss()
    if params['opt'] == 'Adam':
        optimizer = optim.Adam([w], lr=params['lr'])
    elif params['opt'] == 'RMSprop':
        optimizer = optim.RMSprop([w], lr=params['lr'])
    elif params['opt'] == 'SGD':
        optimizer = optim.SGD([w], lr=params['lr'], momentum=params['momentum'])

    for i in range(params['n_steps']):
        optimizer.zero_grad()

        if opt.variable_change:
            x = 0.5 * (torch.tanh(w) + 1.0)
        else:
            x = w.clone()

        loss = loss_f(model2(x), y)
        loss += params['lambda1'] * lp_norm(x, params['alpha'])
        loss += params['lambda2'] * tv_norm(x, params['beta'])

        # print the loss value
        if i % params['print_iter'] == 0:
            loss_np = loss.detach().cpu().numpy()
            print('Iter: {:0>3}, Loss: {}'.format(i, loss_np))

        loss.backward()
        optimizer.step()

        # apply LR decay
        if (i+1) % params['decay_iter'] == 0:
            decay_lr(optimizer, params['decay_factor'])

    y_new = model2(x)[0].detach()
    metric = torch.sum((y_new - y) ** 2) / y.numel()
    return x.detach().cpu(), metric.item()

def optuna_objective(trial, model2, dataloader, device):
    _optimizers = ['SGD', 'Adam', 'RMSprop']

    # parameter
    params = dict()
    params['lambda1'] = 10
    params['lambda2'] = 1
    params['alpha'] = 6
    params['beta'] = 2
    params['n_steps'] = 201
    params['print_iter'] = 200
    params['lr'] = trial.suggest_loguniform('lr', 1e-5, 1e2)
    params['momentum'] = trial.suggest_loguniform('momentum', 1e-3, 1)
    params['decay_iter'] = trial.suggest_int('decay_iter', 10, 100, step=10)
    params['decay_factor'] = trial.suggest_loguniform('decay_factor', 1e-5, 1)
    params['opt'] = trial.suggest_categorical('opt', _optimizers)

    obj = []
    for activations, real_image in dataloader:
        activations = activations.type('torch.FloatTensor').to(device)
        _, v = invert(model2, activations, params, device)
        obj.append(v)
    return np.mean(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_data', type=int, default=16)
    parser.add_argument('--target_layer', type=str, default='22')
    parser.add_argument('--n_trials', type=int, default=100) # used for tuning
    parser.add_argument('--tune_hyperparams', action='store_true')
    parser.add_argument('--variable_change', action='store_true')
    opt = parser.parse_args()

    model_path = 'models/vgg16.pth'
    data_dir = 'data/cifar10/'
    resp_dir = 'resps/vgg16/'
    save_dir = 'generated/vgg16/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda')

    # load the CIFAR10-test images and responses
    data = dst.CIFAR10(data_dir, download=False, train=False,
                       transform=tfs.ToTensor())
    dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
    images = next(iter(dataloader))[0].numpy()
    resp_files = glob(resp_dir +  'test_' + str(opt.target_layer) + '_*.npy')
    for i, file in enumerate(resp_files):
        if i == 0:
            resps = np.load(file)
        else:
            resps = np.vstack((resps, np.load(file)))

    print('image shape: ' + str(images.shape))
    print('resp shape: '  + str(resps.shape))
    test_dataset = CustomDataset(resps[:opt.n_data], images[:opt.n_data])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    model1 = model.VGG()
    model1.to(device)
    model1.load_state_dict(torch.load(model_path))
    model1.eval()
    model2 = model.VGG2(model1, opt.target_layer)
    model2.eval()

    # hyperparameter tuning
    if opt.tune_hyperparams:
        # logging settings
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().addHandler(logging.FileHandler(save_dir + 'log.txt'))
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        # tuning
        study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                    direction='minimize')
        study.optimize(lambda trial: optuna_objective(trial, model2,
                                                      test_loader, device),
                        opt.n_trials)
        results = study.trials_dataframe()

        # save the optimization procedures
        pickle.dump(study, open(save_dir + 'study.pkl', 'wb'), 2)
        results.to_csv(save_dir + 'results.csv')

    # inversion with a fixed hyperparameter set
    params = dict()
    params['lambda1'] = 10
    params['lambda2'] = 1
    params['alpha'] = 6
    params['beta'] = 2
    params['n_steps'] = 201
    params['print_iter'] = 200
    params['lr'] = 1e-2
    params['momentum'] = 0
    params['decay_iter'] = 100
    params['decay_factor'] = 1e-1
    params['opt'] = 'RMSprop' # {'SGD', 'RMSprop', 'Adam'}

    if opt.tune_hyperparams:
        # overwrite `params` if the key exists in the Optuna results
        trial = study.best_trial.number
        cols = results.columns.values.tolist()
        for col in cols:
            if col[:len('params_')] == 'params_':
                params[col[len('params_'):]] = results[col][trial]

    # main inversion
    real_images, new_images = [], []
    for activations, real_image in test_loader:
        activations = activations.type('torch.FloatTensor').to(device)
        new_image, _ = invert(model2, activations, params, device)
        real_images.append(real_image.numpy()[0])
        new_images.append(new_image.numpy()[0])

    # plot the generated images
    plt.figure(figsize=(12, 12))
    for i in range(min(50, opt.n_data)):
        if not opt.variable_change:
            img1 = deprocess_image(real_images[i])
            img2 = deprocess_image(new_images[i])
        else:
            img1 = real_images[i]
            img2 = new_images[i]

        img1 = np.transpose(img1 * 255, (1, 2, 0)).astype('uint8')
        img2 = np.transpose(img2 * 255, (1, 2, 0)).astype('uint8')

        ax = plt.subplot(10, 10, 2 * i + 1)
        ax.imshow(img1)
        ax.set_title('Real')
        plt.axis('off')

        ax = plt.subplot(10, 10, 2 * i + 2)
        ax.imshow(img2)
        ax.set_title('Generated')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir + 'img.png')
    plt.close()
