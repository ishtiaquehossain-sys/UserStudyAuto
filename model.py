import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from importlib import reload
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PyQt5.QtCore import QThread, pyqtSignal
import parameter


class ImgDataset(Dataset):
    def __init__(self, images: list, vectors: list):
        super(ImgDataset, self).__init__()
        reload(parameter)
        param_vector_info = vectors[0].param_vector_info
        self.num_data_points = len(vectors)
        self.images = images
        self.labels = np.array([v.param_vector for v in vectors])
        for i, param in enumerate(param_vector_info):
            min_val = param.min_value
            delta = param.max_value - param.min_value
            self.labels[:, i] = self.labels[:, i] - min_val
            if param.param_type == 's':
                self.labels[:, i] = self.labels[:, i] / delta

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((128, 128))
        ])

    def __getitem__(self, index):
        datum = {'data': self.transform(self.images[index]), 'label': self.labels[index]}
        return datum

    def __len__(self):
        return self.num_data_points


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def conv_layer_set(self, in_c, out_c, k, p, s):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_c),
        )
        return conv_layer

    def fc_layer_set(self, in_c, out_c):
        fc_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(),
        )
        return fc_layer

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureExtractor(BaseModel):
    def __init__(self, in_c):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            self.conv_layer_set(in_c, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            self.conv_layer_set(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            self.conv_layer_set(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            self.conv_layer_set(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            self.conv_layer_set(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            self.conv_layer_set(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.out_c = 256

    def forward(self, x):
        x = self.features(x)
        return x


class TaskModule(BaseModel):
    def __init__(self, task_type, in_c, out_c):
        super(TaskModule, self).__init__()
        self.task_type = task_type
        module = list()
        module.append(self.fc_layer_set(in_c, 64))
        module.append(self.fc_layer_set(64, 64))
        module.append(self.fc_layer_set(64, 16))
        module.append(nn.Linear(16, out_c))
        if task_type == 's':
            module.append(nn.Sigmoid())
        module = nn.Sequential(*module)
        if task_type == 's':
            self.scalar_module = module
        elif task_type == 'i':
            self.integer_module = module
        elif task_type == 'b':
            self.binary_module = module

    def forward(self, x):
        if self.task_type == 's':
            x = self.scalar_module(x)
        elif self.task_type == 'i':
            x = self.integer_module(x)
        elif self.task_type == 'b':
            x = self.binary_module(x)
        return x


class ImgModel(BaseModel):
    def __init__(self, param_vector_info: parameter.ParameterVectorInfo):
        super(ImgModel, self).__init__()
        reload(parameter)
        self.feature_extractor = FeatureExtractor(1)
        in_c = self.feature_extractor.out_c
        self.task_modules = nn.ModuleList()
        for param in param_vector_info:
            if param.param_type == 's':
                self.task_modules.append(TaskModule('s', in_c, 1))
            elif param.param_type == 'i':
                self.task_modules.append(TaskModule('i', in_c, param.max_value-param.min_value+1))
            elif param.param_type == 'b':
                self.task_modules.append(TaskModule('b', in_c, 1))

    def forward(self, x):
        features = self.feature_extractor(x)
        predictions = []
        for module in self.task_modules:
            predictions.append(module(features))
        return predictions


class MultiTaskLoss(nn.Module):
    def __init__(self, param_vector_info: parameter.ParameterVectorInfo):
        super(MultiTaskLoss, self).__init__()
        reload(parameter)
        self.loss_fn = []
        for param in param_vector_info:
            if param.param_type == 's':
                self.loss_fn.append(nn.L1Loss(reduction='sum'))
            elif param.param_type == 'i':
                self.loss_fn.append(nn.CrossEntropyLoss(reduction='sum'))
            elif param.param_type == 'b':
                self.loss_fn.append(nn.BCEWithLogitsLoss(reduction='sum'))

    def forward(self, predictions, targets):
        losses = torch.zeros(len(self.loss_fn))
        losses = losses.to(predictions[0].device)
        for i, loss in enumerate(self.loss_fn):
            losses[i] = loss(predictions[i], targets[i])
        total_loss = torch.sum(losses)
        return losses, total_loss


class Predictor(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self):
        super(Predictor, self).__init__()
        self.shape_name = None
        self.dataset = None
        self.train_size = None
        self.valid_size = None
        self.param_vector_info = None
        self.model = None
        self.is_active = False
        self.is_dirty = False

    def init_model(self, shape_name: str, param_vector_info: parameter.ParameterVectorInfo):
        self.shape_name = shape_name
        self.param_vector_info = param_vector_info
        self.model = ImgModel(param_vector_info)
        try:
            self.model.load_state_dict(torch.load(os.path.join('models', self.shape_name + '.pt')))
            self.model.eval()
        except:
            print('Failed to load ' + self.shape_name + ' model: model does not exist or model definition has changed')
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def init_dataset(self, images: list, vectors: list):
        self.dataset = ImgDataset(images, vectors)

    def __run_epoch(self, optim: torch.optim.Optimizer, loss: MultiTaskLoss, loader: DataLoader, train=True):
        epoch_loss = 0.0
        task_losses = []
        it = iter(loader)
        for batch in tqdm(it, file=sys.stdout):
            if not self.is_active:
                return None, None
            if torch.cuda.is_available():
                batch['data'] = batch['data'].cuda()
                batch['label'] = batch['label'].cuda()
            targets = []
            for i, p in enumerate(self.param_vector_info):
                if p.param_type == 'i':
                    targets.append(batch['label'][:, i].long())
                else:
                    targets.append(batch['label'][:, i:i+1])
            if train:
                self.model.train()
                optim.zero_grad()
                predictions = self.model(batch['data'])
                losses, combined_loss = loss(predictions, targets)
                combined_loss.backward()
                optim.step()
            else:
                self.model.eval()
                predictions = self.model(batch['data'])
                losses, combined_loss = loss(predictions, targets)
            epoch_loss += combined_loss.item()
            task_losses.append(losses.detach().cpu().numpy())
        if train:
            epoch_loss /= self.train_size
            task_losses = np.array(task_losses).sum(axis=0) / self.train_size
        else:
            epoch_loss /= self.valid_size
            task_losses = np.array(task_losses).sum(axis=0) / self.valid_size
        return epoch_loss, task_losses

    def run(self):
        indices = (list(range(len(self.dataset))))
        np.random.shuffle(indices)
        split_point = int(np.floor(0.9 * len(self.dataset)))
        train_indices, valid_indices = indices[:split_point], indices[split_point:]
        self.train_size = len(train_indices)
        self.valid_size = len(valid_indices)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        batch_size = 16
        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
        valid_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
        optim = torch.optim.Adam(self.model.parameters())
        loss = MultiTaskLoss(self.param_vector_info)
        if torch.cuda.is_available():
            loss = loss.cuda()
        num_epochs = 20
        e = 0
        self.is_active = True
        while self.is_active and e < num_epochs:
            train_loss, task_losses = self.__run_epoch(optim, loss, train_loader)
            valid_loss, task_losses = self.__run_epoch(optim, loss, valid_loader, False)
            np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
            print(f'Epoch {e + 1}: T Loss: {train_loss}, V Loss: {valid_loss}')
            print(task_losses, end='')
            print()
            self.progress_signal.emit(int((e+1)*100.0/num_epochs))
            e = e + 1
        if e == num_epochs:
            self.is_dirty = True
            self.save_model()
        self.stop()

    def stop(self):
        self.is_active = False

    def predict(self, index):
        img = Image.open(os.path.join('shapes', self.shape_name, str(index).zfill(2) + '.png')).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((128, 128))
        ])
        img = torch.unsqueeze(transform(img), 0)
        if torch.cuda.is_available():
            img = img.cuda()
        predictions = self.model(img)
        for i, param in enumerate(self.param_vector_info):
            if param.param_type == 'i':
                predictions[i] = F.softmax(predictions[i], dim=1).argmax(-1).unsqueeze(dim=1)
            elif param.param_type == 'b':
                predictions[i] = torch.round(torch.sigmoid(predictions[i]))
            min_val = param.min_value
            delta = param.max_value - param.min_value
            if param.param_type == 's':
                predictions[i] = predictions[i] * delta
            predictions[i] = predictions[i] + min_val
        if predictions:
            param_vector = torch.transpose(torch.cat(predictions), 1, 0).tolist()[0]
        else:
            param_vector = predictions
        return param_vector

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join('models', self.shape_name + '.pt'))
