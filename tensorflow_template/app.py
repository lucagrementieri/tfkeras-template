import logging
import multiprocessing
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from .ingestion import NpyDataset, NpzLoader
from .ingestion.transform import Normalize, Compose
from .models.linear_regression import LinearRegression, linear_regression
from .utils import initialize_logger


# TODO: update class name
class TensorflowTemplate:
    @staticmethod
    def _load_model(checkpoint: str):
        pass

    @staticmethod
    def ingest(root_dir: str, split: str, overwrite: bool = False) -> None:
        initialize_logger()

        # TODO: update transformations
        # TODO: replace values with statistics computed with dataset_statistics.py
        transform = Normalize(
            mean=(0.502, 0.475, 0.475, 0.534, 0.493),
            std=(0.276, 0.270, 0.274, 0.295, 0.299),
        )

        dataset = NpyDataset(root_dir, split, transform=transform)
        split_path = Path(root_dir) / 'npz' / split
        split_path.mkdir(parents=True, exist_ok=overwrite)

        for sample in dataset:
            filename = sample.pop('filename')
            np.savez_compressed(split_path / filename, **sample)

        logging.info('Ingestion completed')

    @staticmethod
    def train(
        npz_dir: str,
        output_dir: str,
        batch_size: int,
        epochs: int,
        lr: float,
        imperative: bool = False,
    ) -> None:
        run_dir = Path(output_dir) / str(int(time.time()))
        checkpoint_dir = run_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        npz_loader = NpzLoader(npz_dir)

        train_dataset = npz_loader.get_split_dataset('train', batch_size)
        dev_dataset = (
            npz_loader.get_split_dataset('dev', batch_size)
            if npz_loader.check_split('dev')
            else None
        )

        # TODO: update scheduler
        decay_steps = 40
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            lr, decay_steps=decay_steps, decay_rate=0.8, staircase=True
        )

        # TODO: update optimizer
        # TODO: remember that state-of-the-art optimizers are included in tfa.optimizers
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=scheduler, momentum=0.9, nesterov=True
        )
        logging.info(f'Optimizer: {optimizer.get_config()}')

        criterion = tf.keras.losses.MeanSquaredError()
        metric = tf.keras.metrics.MeanSquaredError()

        if imperative:
            model = LinearRegression()
        else:
            # TODO: update feature size
            feature_size = 5
            model = linear_regression(feature_size)
        model.compile(optimizer=optimizer, loss=criterion, metrics=[metric])

        checkpoint_path = checkpoint_dir / 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(run_dir / 'logs'), update_freq='epoch', write_graph=False
            ),
        ]
        _ = model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=dev_dataset,
        )

    @staticmethod
    def restore(
        checkpoint: str,
        npz_dir: str,
        output_dir: str,
        batch_size: int,
        epochs: int,
        lr: float,
    ) -> str:
        run_dir = Path(output_dir) / str(int(time.time()))
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
