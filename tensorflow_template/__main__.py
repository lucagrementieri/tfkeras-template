#!/usr/bin/env python3

import argparse
import sys

from tensorflow_template.app import TensorflowTemplate  # TODO: update


class CLI:
    def __init__(self):
        # TODO: update description and usage
        parser = argparse.ArgumentParser(
            description='Command line interface for Tensorflow template',
            usage=(
                'python3 -m tensorflow_template <command> [<args>]\n'
                '\n'
                'ingest      Ingest data\n'
                'train       Train the model\n'
                'eval        Evaluate the model\n'
                'test        Test the model\n'
                'optimize    Optimize the model\n'
            ),
        )
        parser.add_argument(
            'command',
            type=str,
            help='Sub-command to run',
            choices=('ingest', 'train', 'eval', 'test', 'optimize'),
        )

        args = parser.parse_args(sys.argv[1:2])
        command = args.command.replace('-', '_')
        if not hasattr(self, command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, command)()

    @staticmethod
    def ingest() -> None:
        # TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Ingest data',
            usage='python3 -m tensorflow_template ingest data-dir split [--overwrite]',
        )
        # TODO: update parameters and default values
        parser.add_argument(
            'data_dir', metavar='data-dir', type=str, help='Data directory'
        )
        parser.add_argument(
            'split', type=str, help='Split name', choices=('train', 'dev', 'test')
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing TFRecord files if present',
        )

        args = parser.parse_args(sys.argv[2:])
        TensorflowTemplate.ingest(args.data_dir, args.split, args.overwrite)
        print(f'Ingestion completed')

    @staticmethod
    def train() -> None:
        # TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Train the model',
            usage='python3 -m tensorflow_template train npz-dir '
                  '[--output-dir OUTPUT-DIR --batch-size BATCH-SIZE --epochs EPOCHS '
                  '--lr LR --checkpoint CHECKPOINT --imperative]',
        )
        # TODO: update parameters and default values
        parser.add_argument(
            'npz_dir', metavar='npz-dir', type=str, help='Ingested data directory'
        )
        parser.add_argument(
            '--output-dir', type=str, help='Output directory', default='./runs'
        )
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
        parser.add_argument(
            '--lr', type=float, default=0.1, help='Initial learning rate'
        )
        parser.add_argument(
            '--checkpoint', type=str, help='Path to a weight checkpoint'
        )
        parser.add_argument(
            '--imperative', action='store_true', help='Imperative model'
        )

        args = parser.parse_args(sys.argv[2:])
        TensorflowTemplate.train(
            args.npz_dir,
            args.output_dir,
            args.batch_size,
            args.epochs,
            args.lr,
            args.checkpoint,
            args.imperative,
        )

    @staticmethod
    def eval() -> None:
        # TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Evaluate the model',
            usage='python3 -m tensorflow_template eval checkpoint npz-dir '
                  '[--batch-size BATCH-SIZE --imperative]',
        )
        # TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument(
            'npz_dir', metavar='npz-dir', type=str, help='Ingested data directory'
        )
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
        parser.add_argument(
            '--imperative', action='store_true', help='Imperative model'
        )

        args = parser.parse_args(sys.argv[2:])
        val_loss, val_metric = TensorflowTemplate.evaluate(
            args.checkpoint, args.npz_dir, args.batch_size, args.imperative
        )
        print(f'Validation - Loss: {val_loss:.4f} - Metric: {val_metric:.4f}')

    @staticmethod
    def test() -> None:
        # TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Test the model',
            usage='python3 -m tensorflow_template test checkpoint data-path [--imperative]',
        )
        # TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument(
            'data_path', metavar='data-path', type=str, help='Data file path'
        )
        parser.add_argument(
            '--imperative', action='store_true', help='Imperative model'
        )

        args = parser.parse_args(sys.argv[2:])
        prediction = TensorflowTemplate.test(
            args.checkpoint, args.data_path, args.imperative
        )
        print(f'Output: {prediction:.4f}')

    @staticmethod
    def optimize() -> None:
        # TODO: update description coherently with usage in __init__
        parser = argparse.ArgumentParser(
            description='Optimize the model',
            usage='python3 -m tensorflow_template optimize checkpoint [--imperative]',
        )
        # TODO: update parameters and default values
        parser.add_argument('checkpoint', type=str, help='Checkpoint path')
        parser.add_argument(
            '--imperative', action='store_true', help='Imperative model'
        )

        args = parser.parse_args(sys.argv[2:])
        TensorflowTemplate.optimize(args.checkpoint, args.imperative)


if __name__ == '__main__':
    CLI()
