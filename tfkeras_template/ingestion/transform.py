from typing import Any, Callable, Sequence, Dict

import numpy as np


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example['features'] -= self.mean
        example['features'] /= self.std
        return example

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, example: Dict[str, Any]) -> Any:
        for t in self.transforms:
            example = t(example)
        return example

    def __repr__(self) -> str:
        repr_string = self.__class__.__name__ + '(\n'
        for t in self.transforms:
            repr_string += f'    {t}\n'
        repr_string += ')'
        return repr_string
