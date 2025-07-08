import pickle
import lmdb

from torch.utils.data.dataset import Dataset
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["LmdbReader"]


@register_dataset("LmdbReader")
class LmdbReader(DatasetBase):
    r"""
    A reader interface to read datapoints from a serialized LMDB file containing
    ``(image_id, image, caption)`` tuples. Optionally, one may specify a
    partial percentage of datapoints to use.

    .. note::

        When training in distributed setting, make sure each worker has SAME
        random seed because there is some randomness in selecting keys for
        training with partial dataset. If you wish to use a different seed for
        each worker, select keys manually outside of this class and use
        :meth:`set_keys`.

    .. note::

        Similar to :class:`~torch.utils.data.distributed.DistributedSampler`,
        this reader can shuffle the dataset deterministically at the start of
        epoch. Use :meth:`set_shuffle_seed` manually from outside to change the
        seed at every epoch.

    Parameters
    ----------
    lmdb_path: str
        Path to LMDB file with datapoints.
    shuffle: bool, optional (default = True)
        Whether to shuffle or not. If this is on, there will be one deterministic
        shuffle based on epoch before sharding the dataset (to workers).
    percentage: float, optional (default = 100.0)
        Percentage of datapoints to use. If less than 100.0, keys will be
        shuffled and first K% will be retained and use throughout training.
        Make sure to set this only for training, not validation.
    """

    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path

        # fmt: off
        # Create an LMDB transaction right here. It will be aborted when this
        # class goes out of scope.
        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()
        self.cursor = self.db_txn.cursor()

        # Form a list of LMDB keys numbered from 0 (as binary strings).
        self.num_data = env.stat()["entries"]

    def __getstate__(self):
        r"""
        This magic method allows an object of this class to be pickable, useful
        for dataloading with multiple CPU workers. :attr:`db_txn` is not
        pickable, so we remove it from state, and re-instantiate it in
        :meth:`__setstate__`.
        """
        state = self.__dict__
        state["db_txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        env = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()

    def __len__(self):
        return self.num_data

    def __getitem__(self, img_id: str):

        return self.get_data(img_id)

    def get_data(self, img_id: str or list):

        data_instance = None
        if isinstance(img_id, str):
            datapoint_pickled = self.cursor.get(img_id.encode("ascii"))
            data_instance = pickle.loads(datapoint_pickled)

        elif isinstance(img_id, list):
            datapoint_pickled = self.cursor.getmulti([i.encode("ascii") for i in img_id])
            data_instance = [pickle.loads(d[-1]) for d in datapoint_pickled]

        else:
            ValueError("img_id has to be string or list")

        return data_instance
