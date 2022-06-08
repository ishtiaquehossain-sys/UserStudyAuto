import os
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from shapegenerator import ShapeGenerator
from torchvision import models, transforms
from PyQt5.QtCore import QThread, pyqtSignal


class Controller:
    def __init__(self):
        self.cluster_maker = ClusterMaker()

    def remake_images_clusters(self):
        self.cluster_maker.cluster_only = False
        self.cluster_maker.start()

    def remake_clusters(self):
        self.cluster_maker.cluster_only = True
        self.cluster_maker.start()

    def get_summary(self, shape_name: str):
        return self.cluster_maker.get_summary(shape_name)

    def get_details(self, shape_name: str, index: int):
        return self.cluster_maker.get_details(shape_name, index)


class ClusterMaker(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self):
        super(ClusterMaker, self).__init__()
        self.thresh = 50.0
        self.num_clusters = 6
        self.shape_names = ['bench', 'chair']
        self.true_images = {}
        self.pred_images = {}
        self.true_features = {}
        self.pred_features = {}
        self.distances = {}
        self.rearranged_ids = {}
        self.summary_images = {}
        self.cluster_only = False
        self.is_active = False

    def __get_features(self, images: list[Image], feature_extractor: torch.nn.Module):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_batch = torch.stack([transform(ImageOps.invert(image).convert('RGB')) for image in images])
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
        features = feature_extractor(image_batch).detach().cpu().numpy()
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
        if not self.cluster_only:
            generator = ShapeGenerator()
            for shape in self.shape_names:
                self.true_images[shape] = []
                self.pred_images[shape] = []
                for i in range(5):
                    for j in range(10):
                        index = str(i) + str(j)
                        self.true_images[shape].append(
                            Image.open(os.path.join('shapes', shape, index + '.png')).convert('L'))
                        self.pred_images[shape].append(generator.get_image(shape, int(index)))
            feature_extractor = models.alexnet(pretrained=True)
            feature_extractor.classifier = feature_extractor.classifier[:-1]
            if torch.cuda.is_available():
                feature_extractor = feature_extractor.cuda()

            for x, shape in enumerate(self.shape_names):
                self.true_features[shape] = self.__get_features(self.true_images[shape], feature_extractor)
                self.pred_features[shape] = self.__get_features(self.pred_images[shape], feature_extractor)
                self.distances[shape] = self.__get_distances(self.true_features[shape], self.pred_features[shape])
                self.progress_signal.emit(int((x + 1) * 50.0 / len(self.shape_names)))

        color_palette = []
        for i in range(self.num_clusters):
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            color_palette.append((r, g, b))
        for x, shape in enumerate(self.shape_names):
            print('Clustering ' + shape + '...')
            match_idx = sorted(np.where(self.distances[shape] < self.thresh)[0].tolist())
            mismatch_idx = sorted(x for x in range(50) if x not in match_idx)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            kmeans.fit(self.true_features[shape][mismatch_idx].tolist())
            self.rearranged_ids[shape] = match_idx
            colors = [(0, 0, 0)] * len(match_idx)
            for i in range(self.num_clusters):
                cluster = sorted(np.array(mismatch_idx)[np.where(kmeans.labels_ == i)[0].tolist()])
                self.rearranged_ids[shape] = self.rearranged_ids[shape] + cluster
                colors = colors + [color_palette[i]] * len(cluster)

            summary_img = Image.new('RGB', (640, 640))
            for i in range(5):
                for j in range(10):
                    index = i * 10 + j
                    paste_img = self.__colorize_image(
                        self.true_images[shape][self.rearranged_ids[shape][index]],
                        colors[index]
                    ).resize((64, 64))
                    summary_img.paste(paste_img, (j * 64, i * 128))
                    paste_img = self.__colorize_image(
                        self.pred_images[shape][self.rearranged_ids[shape][index]],
                        colors[index]
                    ).resize((64, 64))
                    summary_img.paste(paste_img, (j * 64, 64 + i * 128))
            self.summary_images[shape] = summary_img
            self.progress_signal.emit(int(50.0 + (x + 1) * 50.0 / len(self.shape_names)))

    def get_summary(self, shape_name: str):
        return self.summary_images[shape_name]

    def get_details(self, shape_name: str, index: int):
        return self.true_images[shape_name][self.rearranged_ids[shape_name][index]].convert('RGB'), \
               self.pred_images[shape_name][self.rearranged_ids[shape_name][index]].convert('RGB')

