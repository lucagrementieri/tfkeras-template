import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from .ingestion import NpyDataset, NpyCodec, TFRecordWriter
from .ingestion.transform import Normalize, Compose, Serialize
from .models.linear_regression import LinearRegression, linear_regression
from .utils import initialize_logger


# TODO: update class name
class TensorflowTemplate:
    @staticmethod
    def _load_model(checkpoint: str):
        pass

    @staticmethod
    def ingest(
        root_dir: str,
        split: str,
        records_per_file: int,
        overwrite: bool = False,
        workers: int = 1,
    ) -> None:
        initialize_logger()

        # TODO: update transformations
        transform = Compose(
            [
                # TODO: replace values with statistics computed with dataset_statistics.py
                Normalize(
                    mean=(0.502, 0.475, 0.475, 0.534, 0.493),
                    std=(0.276, 0.270, 0.274, 0.295, 0.299),
                ),
                # TODO: update feature size
                Serialize(NpyCodec(5)),
            ]
        )

        dataset = NpyDataset(root_dir, split, transform=transform)
        split_path = Path(root_dir) / 'tfrecords' / split
        if split_path.exists() and not overwrite:
            raise FileExistsError(f"File exists: '{split_path}'")

        writer = TFRecordWriter(split_path, records_per_file, overwrite, workers)
        writer.start()
        writer.write(dataset)
        writer.close()

        for worker_dir in split_path.iterdir():
            if not worker_dir.is_dir():
                continue
            for source in worker_dir.iterdir():
                source.replace(source.parent.parent / source.name)
            worker_dir.rmdir()

        logging.info('Ingestion completed')

    @staticmethod
    def train(
        tfrecords_dir: str,
        output_dir: str,
        batch_size: int,
        epochs: int,
        lr: float,
        functional: bool = True,
    ) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        tfrecords_dir = Path(tfrecords_dir).expanduser()

        # TODO: update feature size
        feature_size = 5
        codec = NpyCodec(feature_size)

        train_pattern = str(tfrecords_dir / 'train' / '*.tfrecords')
        train_paths = tf.data.Dataset.list_files(train_pattern, shuffle=True)
        train_dataset = tf.data.TFRecordDataset(filenames=train_paths)
        train_dataset = train_dataset.map(
            map_func=codec.decode, num_parallel_calls=os.cpu_count()
        )
        prefetch_size = 128
        shuffle_size = int(1e6)
        train_dataset = train_dataset.shuffle(
            shuffle_size, reshuffle_each_iteration=True
        )
        train_dataset = train_dataset.batch(batch_size=batch_size).prefetch(
            buffer_size=prefetch_size
        )

        if (tfrecords_dir / 'dev').is_dir():
            dev_pattern = str(tfrecords_dir / 'dev' / '*.tfrecords')
            dev_paths = tf.data.Dataset.list_files(dev_pattern, shuffle=True)
            dev_dataset = tf.data.TFRecordDataset(filenames=dev_paths)
            dev_dataset = dev_dataset.map(
                map_func=codec.decode, num_parallel_calls=os.cpu_count()
            )
            prefetch_size = 128
            dev_dataset = dev_dataset.batch(batch_size=batch_size).prefetch(
                buffer_size=prefetch_size
            )
        else:
            dev_dataset = None

        # TODO: update optimizer
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.9, nesterov=True
        )

        # TODO REAL: how to use weight decay
        # logging.info(f'Optimizer: {self._get_optimizer_info(optimizer)}')

        criterion = tf.keras.losses.MeanSquaredError()
        metric = tf.keras.metrics.MeanSquaredError()

        if functional:
            model = linear_regression(feature_size)
            model.compile(optimizer=optimizer, loss=criterion, metrics=[metric])
        else:
            model = LinearRegression()

        history = model.fit(train_dataset, epochs=epochs, validation_data=dev_dataset)

        # return best_checkpoint
        return history

    @staticmethod
    def restore(
        checkpoint: str,
        tensor_dir: str,
        output_dir: str,
        batch_size: int,
        epochs: int,
        lr: float,
    ) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Checkpoint: {checkpoint}')
        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        train_dataset = TorchDataset(tensor_dir, 'train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        if (Path(tensor_dir) / 'dev').is_dir():
            dev_dataset = TorchDataset(tensor_dir, 'dev')
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True,
            )
        else:
            dev_loader = None

        model = TensorflowTemplate._load_model(checkpoint)
        best_checkpoint = model.fit(run_dir, train_loader, epochs, lr, dev_loader)
        return best_checkpoint

    @staticmethod
    def evaluate(
        checkpoint: str, tensor_dir: str, batch_size: int
    ) -> Tuple[float, float]:
        dev_dataset = TorchDataset(tensor_dir, 'dev')
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        model = TensorflowTemplate._load_model(checkpoint)
        val_loss, val_metric = model.eval(dev_loader)
        return val_loss, val_metric

    @staticmethod
    def test(checkpoint: str, data_path: str) -> float:
        initialize_logger()
        model = TensorflowTemplate._load_model(checkpoint)

        # TODO: update transformations to be coherent with what was used during training
        transform = Compose(
            [
                ToTensor(),
                # TODO: if you need normalization, replace values with statistics computed by
                #  dataset_statistics.py ; else remove it.
                Normalize(
                    mean=(0.502, 0.475, 0.475, 0.534, 0.493),
                    std=(0.276, 0.270, 0.274, 0.295, 0.299),
                ),
            ]
        )

        features = {
            'features': transform(torch.load(data_path))
        }  # TODO: update data loading
        prediction = model.predict(features)
        return prediction
