from pathlib import Path
from typing import Any, Optional, Callable, Dict

import numpy as np
import pandas as pd

from .iterable_dataset import IterableDataset


# TODO update dataset class
class NpyDataset(IterableDataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.transform = transform

        csv_paths = [p for p in self.root_dir.iterdir() if p.suffix == '.csv']
        if len(csv_paths) > 1:
            raise IOError('Several csv files in the given folder')
        self.dataframe = pd.read_csv(csv_paths[0], index_col=0)
        self.dataframe = self.dataframe.filter(regex=f'^{split}', axis=0)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Any:
        # TODO: update return types
        filepath = self.root_dir / self.dataframe.index[idx]
        # TODO: update
        example = {
            'features': np.load(filepath),
            'target': self.dataframe.iloc[idx, 0],
            'filename': filepath.stem,
        }

        if self.transform:
            example = self.transform(example)

        return example
