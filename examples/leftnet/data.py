# make data to fit the network
from torch.utils.data import Dataset

from khot_embeddings import KHOT_EMBEDDINGS
from qmof_khot_embeddings import QMOF_KHOT_EMBEDDINGS

import torch

class Data:
    def __init__(self, dataset_name, *args, **kwargs):
        ''' dataset_name: str, name of the dataset, 'omdb', 'qmof' '''
        self.dataset_name = dataset_name
        # if arg length not 3, raise error
        if len(args) != 3:
            raise ValueError('args length must be 3')
        # if arg length is 3, set the attributes
        self.set0 = args[0]
        self.set1 = args[1]
        self.set2 = args[2]

    def join_dataset(self):
        joined = self.set0.merge(self.set1, on='id', how='left')
        joined = joined.merge(self.set2, on='id', how='left')

        return joined

def get_data(dataset):
    embeddings = KHOT_EMBEDDINGS
    embedding = torch.zeros(100, len(embeddings[1]))
    for i in range(100):
        embedding[i] = torch.tensor(embeddings[i + 1])
    # embedding_fc = torch.nn.Linear(len(embeddings[1]), cfg.MODEL.ATOM_EMBEDDING_SIZE)
    # for each item in the dict dataset, set the x attribute to the embedding
    list_x = []
    list_y = []
    for data_ in dataset:

        data_.x = embedding[data_.atomic_numbers.long() - 1]
        list_x.append(data_.x)
        list_y.append(data_.y)
    return list_x, list_y

class JoinedLmdbDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = self.cumulative_size(self.datasets)

    def cumulative_size(self, datasets):
        return [sum(d.num_samples for d in datasets[:i+1]) for i in range(len(datasets))]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = next(i for i, cumulative_size in enumerate(self.cumulative_sizes) if cumulative_size > idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.cumulative_sizes[-1]


#
# class DatasetWithEmbedding(Dataset):
#     def __init__(self, dataset, embeddings, y_class=False, y_class_value=None, y_class_threshold=None):
#         self.dataset = [data for data in dataset if y_class is False or abs(data.y - y_class_value) > y_class_threshold]
#         self.embeddings = embeddings
#         self.y_class = y_class
#         self.y_class_value = y_class_value
#         self.y_class_threshold = y_class_threshold
#
#     def __getitem__(self, idx):
#         data_ = self.dataset[idx]
#         print(self.embeddings.keys())
#         data_.x = self.embeddings[data_.atomic_numbers.long()]
#         if self.y_class:
#             if data_.y < self.y_class_value:
#                 data_.y = 0
#             else:
#                 data_.y = 1
#         return data_.x, data_.y
#
#     def __len__(self):
#         return len(self.dataset)