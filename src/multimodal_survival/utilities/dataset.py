from typing import Any, Tuple

import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset


def get_datasets(train_filepath: str, test_filepath: str, pipeline: Pipeline) -> Tuple:
    """Creates an instance of OmicsDataset.

    Args:
        train_filepath: Path to training data csv.
        test_filepath: Path to testing data csv.
        pipeline: Instance of sklearn.pipeline to process data.

    Returns:
        Training and testing dataset instances from OmicsDataset.
    """
    train_data = pd.read_csv(train_filepath, index_col=0)
    test_data = pd.read_csv(test_filepath, index_col=0)
    train_dataset = OmicsDataset(train_data, pipeline)
    test_dataset = OmicsDataset(
        test_data, train_dataset.preprocessing_pipeline, eval=True
    )

    return train_dataset, test_dataset


class OmicsDataset(Dataset):
    def __init__(self, data: ArrayLike, pipeline: Pipeline, eval: bool = False) -> None:
        """Constructor.

        Args:
            data: Data to sample.
            pipeline: Pipeline to process the data.
            eval: Whether or not creating an instance of test data, has implications for the pipeline. Defaults to False.
        """
        super().__init__()

        self.dataset = data
        self.preprocessing_pipeline = pipeline
        if eval:
            self.dataset = self.preprocessing_pipeline.transform(self.dataset)
        else:
            self.dataset = self.preprocessing_pipeline.fit_transform(self.dataset)

        self.dataset = torch.as_tensor(self.dataset, dtype=torch.float32)

    def __len__(self):
        """Get length of dataset.

        Returns:
            length of dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        """Get item.

        Args:
            index: Index to retrieve.

        Returns:
            Single item from dataset.
        """
        # convert index to list
        # index = list(index)

        return self.dataset[index, :]
