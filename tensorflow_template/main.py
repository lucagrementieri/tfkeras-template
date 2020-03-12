import json
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Tuple

import torch

from .ingestion import NpyDataset, NpyCodec, TFRecordWriter
from .ingestion.transform import Normalize, Compose, Serialize
from .models.linear import LinearRegression
from .models.model import Model
from .utils import initialize_logger


# TODO: update class name
class TensorflowTemplate:
    @staticmethod
    def _load_model(checkpoint: str) -> Model:
        with open(Path(checkpoint).parent.parent / 'hyperparams.json', 'r') as f:
            hyperparams = json.load(f)

        # TODO: update module
        if LinearRegression.__name__ == hyperparams['module_name']:
            module_class = LinearRegression  # TODO: update module
        else:
            raise ValueError('Checkpoint of unsupported module')
        del hyperparams['module_name']

        module = module_class(**hyperparams)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        module.load_state_dict(torch.load(checkpoint, map_location=device))
        model = Model(module)
        return model

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
        tensor_dir: str, output_dir: str, batch_size: int, epochs: int, lr: float
    ) -> str:
        run_dir = Path(output_dir) / 'runs' / str(int(time.time()))
        (run_dir / 'checkpoints').mkdir(parents=True)
        initialize_logger(run_dir)

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

        module = LinearRegression(train_dataset.features_shape[-1])
        model = Model(module)
        best_checkpoint = model.fit(run_dir, train_loader, epochs, lr, dev_loader)
        return best_checkpoint

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
