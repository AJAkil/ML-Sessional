import numpy as np
import gzip
import os
from sklearn import preprocessing

class MnistDataLoader:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.train_file_name = 'train-images-idx3-ubyte.gz'
        self.train_label_file_name = 'train-labels-idx1-ubyte.gz'
        self.test_file_name = 't10k-images-idx3-ubyte.gz'
        self.test_label_file_name = 't10k-labels-idx1-ubyte.gz'
        self.data = dict()
        self.size = 28
        self.color_channel = 1
        self.data_list = [
            'train_images',
            'train_labels',
            'test_images',
            'test_labels'
        ]

    def load_images(self, data_list_index, file_name):
        images = gzip.open(os.path.join(self.data_folder_path, file_name), 'rb')
        self.data[self.data_list[data_list_index]] = np.frombuffer(images.read(), dtype=np.uint8, offset=16).reshape(-1, self.size, self.size)
        self.data[self.data_list[data_list_index]] = self.data[self.data_list[data_list_index]].reshape(self.data[self.data_list[data_list_index]].shape[0], self.size, self.size, self.color_channel).astype(np.float32)

    def load_labels(self, data_list_index, file_name):
        labels = gzip.open(os.path.join(self.data_folder_path, file_name), 'rb')
        self.data[self.data_list[data_list_index]] = np.frombuffer(labels.read(), dtype=np.uint8, offset=8)
        self.data[self.data_list[data_list_index]].resize(self.data[self.data_list[data_list_index]].shape[0],1)

    def load_mnist(self):
        self.load_images(data_list_index=0, file_name=self.train_file_name)
        self.load_labels(data_list_index=1, file_name=self.train_label_file_name)
        self.load_images(data_list_index=2, file_name=self.test_file_name)
        self.load_labels(data_list_index=3, file_name=self.test_label_file_name)

        self.assert_data_shape()

    def assert_data_shape(self):
        assert self.data[self.data_list[0]].shape == (60000, 28, 28, 1)
        assert self.data[self.data_list[1]].shape == (60000, 1)
        assert self.data[self.data_list[2]].shape == (10000, 28, 28, 1)
        assert self.data[self.data_list[3]].shape == (10000, 1)

    def preprocess_data(self):

        self.data[self.data_list[0]] /= 255
        self.data[self.data_list[2]] /= 255

        self.data[self.data_list[1]] = Utility.one_hot_encode(self.data[self.data_list[1]])

        assert self.data[self.data_list[1]].shape == (60000, 10)
