import logging
import pathlib
import shutil
from multiprocessing import Process, Queue
from typing import Union

import tensorflow as tf

from .iterable_dataset import IterableDataset


class TFRecordWriterProcess(Process):
    END_QUEUE = 0

    def __init__(
        self,
        queue: Queue,
        path: Union[str, pathlib.Path],
        records_per_file: int,
        overwrite: bool = False,
    ):
        super().__init__(target=self.write, args=(queue,))
        self.path = pathlib.Path(path).expanduser()
        self.records_per_file = records_per_file
        if overwrite and self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)

        self.record_counter = 0
        self.file_counter = -1
        self.file = None

    def __next_file(self) -> None:
        self.close()
        self.file_counter += 1
        file_path = str(
            self.path / f'{self.path.name}_{self.file_counter:04d}.tfrecords'
        )
        self.file = tf.io.TFRecordWriter(file_path)
        logging.info(f'Opened file at {file_path}')

    def write(self, q: Queue) -> None:
        while True:
            dataset = q.get()
            if dataset == self.END_QUEUE:
                logging.info(f'Close {self.path} after {self.record_counter} records')
                self.close()
                return
            for example in dataset:
                if self.record_counter % self.records_per_file == 0:
                    self.__next_file()
                if self.record_counter % 100000 == 0:
                    logging.info(
                        f'Records written to {self.path}: {self.record_counter}'
                    )
                self.file.write(example)
                self.record_counter += 1

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class TFRecordWriter:
    def __init__(
        self,
        path: Union[str, pathlib.Path],
        records_per_file: int,
        overwrite: bool = False,
        processes: int = 1,
    ):
        path = pathlib.Path(path).expanduser()
        self.queue = Queue()
        self.processes = [
            TFRecordWriterProcess(
                self.queue, path / f'{i:02d}', records_per_file, overwrite
            )
            for i in range(processes)
        ]

    def start(self) -> None:
        for process in self.processes:
            process.start()

    def write(self, dataset: IterableDataset) -> None:
        self.queue.put(dataset)

    def close(self) -> None:
        for process in self.processes:
            self.queue.put(process.END_QUEUE)
        for process in self.processes:
            process.join()
