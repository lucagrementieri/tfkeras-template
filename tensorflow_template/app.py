import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.python.framework.convert_to_constants as conversion

from .ingestion import NpyDataset, NpzLoader
from .ingestion.transform import Normalize
from .models.linear_regression import LinearRegression, linear_regression
from .utils import initialize_logger


# TODO: update class name
class TensorflowTemplate:
    @staticmethod
    def build_model(
            feature_size: Optional[int] = None,
            checkpoint: Optional[str] = None,
            imperative: bool = False,
    ) -> tf.keras.Model:
        # TODO update model
        model = LinearRegression() if imperative else linear_regression(feature_size)
        if checkpoint is not None:
            model.load_weights(checkpoint).expect_partial()
        return model

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
            checkpoint: Optional[str] = None,
            imperative: bool = False,
    ) -> None:
        model_type = 'imperative' if imperative else 'symbolic'
        run_dir = Path(output_dir) / f'{int(time.time())}-{model_type}'
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
        # TODO: update feature size
        feature_size = 5
        model = TensorflowTemplate.build_model(feature_size, checkpoint, imperative)
        model.compile(optimizer=optimizer, loss=criterion, metrics=[metric])

        checkpoint_path = checkpoint_dir / 'checkpoint-{epoch:02d}-{val_loss:.2f}'
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
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
    def evaluate(
            checkpoint: str, npz_dir: str, batch_size: int, imperative: bool = False
    ) -> Tuple[float, float]:
        npz_loader = NpzLoader(npz_dir)
        dev_dataset = npz_loader.get_split_dataset('dev', batch_size)

        # TODO: update feature size
        feature_size = 5
        model = TensorflowTemplate.build_model(feature_size, checkpoint, imperative)
        criterion = tf.keras.losses.MeanSquaredError()
        metric = tf.keras.metrics.MeanSquaredError()
        model.compile(loss=criterion, metrics=[metric])
        results = model.evaluate(dev_dataset)
        return tuple(results)

    @staticmethod
    def test(checkpoint: str, data_path: str, imperative: bool = False) -> float:
        initialize_logger()
        # TODO: update feature size
        feature_size = 5
        model = TensorflowTemplate.build_model(feature_size, checkpoint, imperative)

        # TODO: update transformations to be coherent with what was used during training
        transform = Normalize(
            mean=(0.502, 0.475, 0.475, 0.534, 0.493),
            std=(0.276, 0.270, 0.274, 0.295, 0.299),
        )
        example = transform({'features': np.expand_dims(np.load(data_path), axis=0)})
        prediction = model.predict(example['features'], batch_size=1)
        return float(prediction[0])

    @staticmethod
    def optimize(checkpoint: str, imperative: bool = False):
        # TODO: update feature size
        feature_size = 5
        model = TensorflowTemplate.build_model(feature_size, checkpoint, imperative)
        options = {
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'disable_model_pruning': False,
            'scoped_allocator_optimization': True,
            'pin_to_host_optimization': True,
            'implementation_selector': True,
            'disable_meta_optimizer': False,
        }
        tf.config.optimizer.set_experimental_options(options)

        model_func = tf.function(model)
        x = tf.zeros((1, feature_size), dtype=tf.float32)
        model_func(x)  # perform tracing
        concrete_func = model_func.get_concrete_function(
            tf.TensorSpec((1, feature_size), dtype=tf.float32)
        )
        print(concrete_func.graph.get_operations())

        frozen_func = conversion.convert_variables_to_constants_v2(concrete_func)
        frozen_func(x)  # perform tracing
        frozen_func(np.array([[0, 1, 0, 0, 0]]))
        return frozen_func