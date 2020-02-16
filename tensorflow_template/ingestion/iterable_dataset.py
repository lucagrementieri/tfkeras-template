from abc import ABC, abstractmethod
from typing import Any, Iterator


class IterableDataset(ABC):
    def __init__(self):
        self.__idx = None

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __next__(self) -> Any:
        if self.__idx < len(self):
            example = self[self.__idx]
            self.__idx += 1
            return example
        raise StopIteration

    def __iter__(self) -> Iterator:
        self.__idx = 0
        return self
