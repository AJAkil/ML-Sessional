import numpy as np
import gzip
import os
import pickle
from sklearn import preprocessing

class Utility:

    @staticmethod
    def one_hot_encode(y_true):
        # Define the One-hot Encoder
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(y_true)
        y_true = ohe.transform(y_true).toarray()
        return y_true

    @staticmethod
    def zero_pad(tensor, pad_size):
        """
        :param tensor: tensor of shape (h, w, num_channel)
        :return: padded tensor of shape (h + 2 * pad_size, w + 2 * pad_size, num_channel)
        """
        return np.pad(tensor, ((pad_size, pad_size), (pad_size, pad_size), (0,0)), mode='constant', constant_values=0)

    @staticmethod
    def convolve_single_step(Z_prev_windowed, W, b):
        """
        :param Z_prev_windowed: window of shape (F, F, num_channel_Z_prev)
        :param W: kernel/filter/weight of shape (F, F, num_channel_Z_prev)
        :param b: bias term of shape (1, 1, 1)
        :return: scaler convolved value
        """
        return np.multiply(Z_prev_windowed, W).sum() + float(b)


    @staticmethod
    def get_max_pool_window(Z_prev_windowed):
        return Z_prev_windowed.max()

    def create_mini_batches(self):
        pass

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


class Cifer10DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.size = 32
        self.color_channel = 3
        self.per_batch_data_size = 10000
        self.data = dict()

    def load_data(self, file_name):
        with open(os.path.join(self.data_path, file_name), 'rb') as f:

            data_dict=pickle.load(f, encoding='latin1')

            images = data_dict['data']
            labels = data_dict['labels']

            images = images.reshape(self.per_batch_data_size, self.color_channel, self.size, self.size).transpose(0,2,3,1).astype("float")
            labels = np.array(labels)
            print(labels.shape)

            return images, labels

    def concatenate_data(self):
        X1, Y1 = self.load_data('data_batch_1')
        X2, Y2 = self.load_data('data_batch_2')
        X3, Y3 = self.load_data('data_batch_3')
        X4, Y4 = self.load_data('data_batch_4')
        X5, Y5 = self.load_data('data_batch_5')

        self.data['train_images'] = np.concatenate(
            (
                X1, X2, X3, X4, X5
            ),
            axis=0
        )

        self.data['train_labels'] = np.concatenate(
            (
                Y1.reshape(self.per_batch_data_size, 1),
                Y2.reshape(self.per_batch_data_size, 1),
                Y3.reshape(self.per_batch_data_size, 1),
                Y4.reshape(self.per_batch_data_size, 1),
                Y5.reshape(self.per_batch_data_size, 1)
            ),
            axis=0
        )

        X_test, Y_test = self.load_data('test_batch')

        self.data['test_images'] = X_test
        self.data['test_labels'] = Y_test.reshape(Y_test.shape[0], 1)

        self.assert_data_shape()

        for key, data in self.data.items():
            print(f'Shape: {data.shape}')

    def assert_data_shape(self):
        assert self.data['train_images'].shape == (50000, 32, 32, 3)
        assert self.data['train_labels'].shape == (50000, 1)
        assert self.data['test_images'].shape  == (10000, 32, 32, 3)
        assert self.data['test_labels'].shape  == (10000, 1)

    def preprocess_data(self):
        self.data['train_images'] /= 255
        self.data['test_images'] /= 255

        self.data['train_labels'] = Utility.one_hot_encode(self.data['train_labels'])

        assert self.data['train_labels'].shape == (50000, 10)


