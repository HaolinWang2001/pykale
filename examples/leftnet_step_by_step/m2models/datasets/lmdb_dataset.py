"""
Codes borrowed from Open Catalyst Project (OCP) https://github.com/Open-Catalyst-Project/ocp (MIT license)
"""

import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch_geometric
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from m2models.common import distutils
from m2models.common.registry import registry
from m2models.common.utils import pyg2_data_transform
from torch_geometric.data import Batch

from examples.leftnet_step_by_step.m2models.common.utils import radius_graph_pbc, get_pbc_distances


# from m2models.common.data_parallel import (
#     # BalancedBatchSampler,
#     # OCPDataParallel,
#     ParallelCollater,
# )


@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")
class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(LmdbDataset, self).__init__()
        self.config = config


        # assert not self.config.get(
        #     "train_on_oc20_total_energies", False
        # ), "For training on total energies set dataset=oc22_lmdb"

        self.path = Path(self.config["src"])

        # self.train_path = Path(self.config["dataset"]["src"])
        # self.val_path = Path(self.config["val_dataset"]["src"])
        # self.test_path = Path(self.config["test_dataset"]["src"])

        # if not self.path.is_file():
        db_paths = sorted(self.path.glob("*.lmdb"))
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

        # self.load_data()

        # else:
        #     self.metadata_path = self.path.parent / "metadata.npz"
        #     self.env = self.connect_db(self.path)
        #     self._keys = [
        #         f"{j}".encode("ascii")
        #         for j in range(self.env.stat()["entries"])
        #     ]
        #     self.num_samples = len(self._keys)

        # # If specified, limit dataset to only a portion of the entire dataset
        # # total_shards: defines total chunks to partition dataset
        # # shard: defines dataset shard to make visible
        #
        #
        # self.sharded = False
        # if "shard" in self.config and "total_shards" in self.config:
        #     self.sharded = True
        #     self.indices = range(self.num_samples)
        #     # split all available indices into 'total_shards' bins
        #     self.shards = np.array_split(
        #         self.indices, self.config.get("total_shards", 1)
        #     )
        #     # limit each process to see a subset of data based off defined shard
        #     self.available_indices = self.shards[self.config.get("shard", 0)]
        #     self.num_samples = len(self.available_indices)
        #
        # self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
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
        data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
        data_object.id = f"{db_idx}_{el_idx}"



        # if self.transform is not None:
        #     data_object = self.transform(data_object)

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

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()



# class SinglePointLmdbDataset(LmdbDataset):
#     def __init__(self, config, transform=None):
#         super(SinglePointLmdbDataset, self).__init__(config, transform)
#         warnings.warn(
#             "SinglePointLmdbDataset is deprecated and will be removed in the future."
#             "Please use 'LmdbDataset' instead.",
#             stacklevel=3,
#         )


# class TrajectoryLmdbDataset(LmdbDataset):
#     def __init__(self, config, transform=None):
#         super(TrajectoryLmdbDataset, self).__init__(config, transform)
#         warnings.warn(
#             "TrajectoryLmdbDataset is deprecated and will be removed in the future."
#             "Please use 'LmdbDataset' instead.",
#             stacklevel=3,
#         )


def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch

def generate_graph(
    data,
    cutoff=None,
    max_neighbors=None,
    use_pbc=None,
    otf_graph=None,
):
    cutoff = cutoff
    max_neighbors = max_neighbors
    use_pbc = use_pbc
    otf_graph = otf_graph

    if not otf_graph:
        try:
            edge_index = data.edge_index

            if use_pbc:
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

        except AttributeError:
            logging.warning(
                "Turning otf_graph=True as required attributes not present in data object"
            )
            otf_graph = True

    if use_pbc:
        if otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, cutoff, max_neighbors
            )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        if otf_graph:
            edge_index = radius_graph(
                data.pos,
                r=cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(
            edge_index.shape[1], 3, device=data.pos.device
        )
        cell_offset_distances = torch.zeros_like(
            cell_offsets, device=data.pos.device
        )
        neighbors = compute_neighbors(data, edge_index)

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )
