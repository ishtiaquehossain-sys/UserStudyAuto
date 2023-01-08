import os
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import shapegenerator
from importlib import reload
from torchvision import models, transforms
from PyQt5.QtCore import QThread, pyqtSignal


class ShapeSorter(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self):
        super(ShapeSorter, self).__init__()
        self.shape_names = ['bench', 'table']
        self.thresh = 30.0
        self.num_clusters = 6
        self.true_images = {}
        self.pred_images = {}
        self.true_features = {}
        self.pred_features = {}
        self.rearranged_ids = {}
        self.summary_images = {}
        self.generator = None
        self.clusters = None

        self.feature_extractor = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.feature_extractor.eval()
        self.feature_extractor.classifier = self.feature_extractor.classifier[:-1]
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()

    def __get_features(self, images: list[Image]):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_batch = torch.stack([transform(ImageOps.invert(image).convert('RGB')) for image in images])
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
        features = self.feature_extractor(image_batch).detach().cpu().numpy()
        for i, image in enumerate(images):
            image = np.array(image.getdata()).reshape(image.size[0], image.size[1])
            image = np.ones_like(image) * 255 - image
            idx = np.where(image != 0)
            if len(idx[0]) == 0 or len(idx[1]) == 0:
                features[i, :] = np.inf
        return features

    def __get_distances(self, features1, features2):
        distances = []
        for i in range(len(features1)):
            distances.append(np.linalg.norm(features1[i] - features2[i]))
        return np.array(distances)

    def __colorize_image(self, image: Image, color: tuple):
        data = image.getdata()
        new_data = []
        for px in data:
            if px == 0:
                new_data.append(color)
            else:
                new_data.append((255, 255, 255))
        c_image = Image.new('RGB', (image.size[0], image.size[1]))
        c_image.putdata(new_data)
        return c_image

    def run(self):
        reload(shapegenerator)
        self.generator = shapegenerator.ShapeGenerator()
        for x, shape in enumerate(self.shape_names):
            self.true_images[shape] = []
            self.pred_images[shape] = []
            for i in range(5):
                for j in range(10):
                    index = str(i) + str(j)
                    self.true_images[shape].append(
                        Image.open(os.path.join('shapes', shape, index + '.png')).convert('L'))
                    self.pred_images[shape].append(self.generator.get_image(shape, int(index)))
            self.progress_signal.emit(int((x + 1) * 25.0 / len(self.shape_names)))

        distances = {}
        for x, shape in enumerate(self.shape_names):
            self.true_features[shape] = self.__get_features(self.true_images[shape])
            self.pred_features[shape] = self.__get_features(self.pred_images[shape])
            distances[shape] = self.__get_distances(self.true_features[shape], self.pred_features[shape])
            self.progress_signal.emit(int(25.0 + (x + 1) * 25.0 / len(self.shape_names)))

        clusters = {}
        for x, shape in enumerate(self.shape_names):
            print('Clustering ' + shape + '...')
            match_idx = sorted(np.where(distances[shape] < self.thresh)[0].tolist())
            mismatch_idx = sorted(x for x in range(50) if x not in match_idx)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            kmeans.fit(self.true_features[shape][mismatch_idx].tolist())
            clustered_mismatch_idx = []
            for i in range(self.num_clusters):
                cluster = sorted(np.array(mismatch_idx)[np.where(kmeans.labels_ == i)[0].tolist()])
                clustered_mismatch_idx = clustered_mismatch_idx + cluster
            clusters[shape] = [match_idx, clustered_mismatch_idx]
            self.progress_signal.emit(int(50.0 + (x + 1) * 25.0 / len(self.shape_names)))

        for x, shape in enumerate(self.shape_names):
            self.rearranged_ids[shape] = clusters[shape][0] + clusters[shape][1]
            colors = [(34, 139, 34)] * len(clusters[shape][0]) + [(0, 0, 0)] * len(clusters[shape][1])
            summary_img = Image.new('RGBA', (680, 640))
            for i in range(5):
                for j in range(10):
                    index = i * 10 + j
                    paste_img = self.__colorize_image(
                        self.true_images[shape][self.rearranged_ids[shape][index]],
                        colors[index]
                    ).resize((64, 64))
                    summary_img.paste(paste_img, (i * 138, j * 64))
                    paste_img = self.__colorize_image(
                        self.pred_images[shape][self.rearranged_ids[shape][index]],
                        colors[index]
                    ).resize((64, 64))
                    summary_img.paste(paste_img, (64 + i * 138, j * 64))
            self.summary_images[shape] = summary_img
            self.progress_signal.emit(int(75.0 + (x + 1) * 25.0 / len(self.shape_names)))

    def get_summary(self, shape_name: str):
        return self.summary_images[shape_name]

    def get_details(self, shape_name: str, index: int):
        return self.true_images[shape_name][self.rearranged_ids[shape_name][index]].convert('RGB'), \
               self.pred_images[shape_name][self.rearranged_ids[shape_name][index]].convert('RGB')
