import os
import pickle
from pathlib import Path
import lmdb
import numpy as np
import bisect
import torch_geometric
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data

class LmdbDataset:
    def __init__(self, config, transform=None):
        self.config = config
        self.path = Path(self.config["src"])
        db_paths = sorted(self.path.glob("*.lmdb"))

        print("db_paths: ", os.listdir(self.path))

        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

        self.metadata_path = self.path / "metadata.npz"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(
                self.envs[-1].begin().get("length".encode("ascii"))
            )
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect_right(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object =pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
                str(lmdb_path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
        )
        return env

def pyg2_data_transform(data: Data):
    """
    if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    we need to convert the data to the new format
    """
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return Data(**{k: v for k, v in data.__dict__.items() if v is not None})
    return data

def close_db(self):
    if not self.path.is_file():
        for env in self.envs:
            env.close()
    else:
        self.env.close()



def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                for j in range(data.natoms.shape[0]):
                    n_index = data[j].edge_index[1, :]
                    n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch