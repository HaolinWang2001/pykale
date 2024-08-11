from kale.loaddata.dataset_access import DatasetAccess
from m2models.datasets.lmdb_dataset import LmdbDataset
from m2models.datasets.lmdb_dataset import generate_graph


class LmdbDatasetAccess(DatasetAccess):
    '''
    Common API for LMDB dataset access
    Args:
        data_path (str): Path to the LMDB dataset
        seed (int):
    '''
    def __init__(self, config_train, config_val, config_test):
        self.config_train = config_train
        self.config_val = config_val
        self.config_test = config_test
        # self._seed = seed

    def get_train(self):
        train_dataset = LmdbDataset(self.config_train)
        # for sample in train_dataset:
        #     (
        #         edge_index,
        #         distances,
        #         distance_vec,
        #         cell_offsets,
        #         _,
        #         neighbors,
        #     ) = generate_graph(
        #         sample,
        #         cutoff=6.0,
        #         max_neighbors=50,
        #         use_pbc=True,
        #         otf_graph=None,
        #     )
        return train_dataset

    def get_valid(self):
        valid_dataset = LmdbDataset(self.config_val)
        return valid_dataset

    def get_test(self):
        test_dataset = LmdbDataset(self.config_test)
        return test_dataset

