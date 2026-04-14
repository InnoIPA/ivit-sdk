#!/usr/bin/env python3
"""
iVIT-SDK AI Application Developer Training Example

Target: AI developers who need to train and deploy custom models
        using transfer learning.

Features demonstrated:
- Dataset preparation (ImageFolder format)
- Transfer learning with pre-trained models
- Training callbacks (EarlyStopping, ModelCheckpoint)
- Model evaluation and export

Usage:
    # Prepare your dataset in ImageFolder format:
    # my_dataset/
    #   cat/
    #     image1.jpg
    #     image2.jpg
    #   dog/
    #     image1.jpg
    #     image2.jpg

    python ai_developer_training.py --dataset ./my_dataset
    python ai_developer_training.py --dataset ./my_dataset --model efficientnet-b0 --epochs 30
"""

import argparse
import os
import sys

import ivit
from ivit.train import (
    Trainer,
    ImageFolderDataset,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    LRScheduler,
)


def prepare_datasets(dataset_path: str, train_split: float = 0.8):
    """Step 1: Prepare training and validation datasets."""
    print("=" * 60)
    print("Step 1: Dataset Preparation")
    print("=" * 60)

    # Create training dataset
    train_dataset = ImageFolderDataset(
        root=dataset_path,
        train_split=train_split,
        split="train"
    )

    # Create validation dataset
    val_dataset = ImageFolderDataset(
        root=dataset_path,
        train_split=train_split,
        split="val"
    )

    print(f"Dataset path: {dataset_path}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class names: {train_dataset.class_names}")

    return train_dataset, val_dataset


def create_trainer(
    model_name: str,
    train_dataset,
    val_dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    freeze_backbone: bool = True,
):
    """Step 2: Create and configure trainer."""
    print("\n" + "=" * 60)
    print("Step 2: Trainer Configuration")
    print("=" * 60)

    trainer = Trainer(
        model=model_name,
        dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        freeze_backbone=freeze_backbone,
        optimizer="adam",
    )

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze backbone: {freeze_backbone} (transfer learning)")
    print(f"Optimizer: adam")

    print("\nSupported pre-trained models:")
    print("  ResNet: resnet18, resnet34, resnet50, resnet101")
    print("  EfficientNet: efficientnet-b0, efficientnet-b1, efficientnet-b2")
    print("  MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large")
    print("  VGG: vgg16, vgg19")
    print("  DenseNet: densenet121")

    return trainer


def setup_callbacks(output_dir: str):
    """Step 3: Setup training callbacks."""
    print("\n" + "=" * 60)
    print("Step 3: Training Callbacks")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    callbacks = [
        # Early stopping: stop when validation loss doesn't improve
        EarlyStopping(
            patience=5,
            monitor="val_loss",
            min_delta=0.001
        ),

        # Model checkpoint: save best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.pt"),
            monitor="val_accuracy",
            save_best_only=True
        ),

        # Progress logger: print training progress
        ProgressLogger(),

        # Learning rate scheduler: reduce LR on plateau
        LRScheduler(
            schedule_type="step",
            step_size=10,
            gamma=0.1
        ),
    ]

    print("Configured callbacks:")
    print("  - EarlyStopping: patience=5, monitor=val_loss")
    print("  - ModelCheckpoint: save best model by val_accuracy")
    print("  - ProgressLogger: print training progress")
    print("  - LRScheduler: step decay, step_size=10, gamma=0.1")

    return callbacks


def train_model(trainer, callbacks):
    """Step 4: Execute training."""
    print("\n" + "=" * 60)
    print("Step 4: Training")
    print("=" * 60)

    print("Starting training...")
    history = trainer.fit(callbacks=callbacks)
    print(history)
    print("\nTraining completed!")
    if history and isinstance(history[-1], dict):
        last = history[-1]
        print(f"Final training loss: {last['loss']:.4f}")
        print(f"Final training accuracy: {last['accuracy']:.2f}%")
        print(f"Final validation loss: {last['val_loss']:.4f}")
        print(f"Final validation accuracy: {last['val_accuracy']:.2f}%")
        # print(f"Best validation accuracy: {max(last['val_accuracy']):.2%}")

    else:
        print("Training history format is unexpected.")
        
    return history


def evaluate_model(trainer):
    """Step 5: Evaluate model."""
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    metrics = trainer.evaluate()

    print("Evaluation results:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Loss: {metrics['loss']:.4f}")

    if 'precision' in metrics:
        print(f"  Precision: {metrics['precision']:.2%}")
    if 'recall' in metrics:
        print(f"  Recall: {metrics['recall']:.2%}")
    if 'f1' in metrics:
        print(f"  F1 Score: {metrics['f1']:.2%}")

    return metrics


def export_model(trainer, output_dir: str, quantize: str = "fp16"):
    """Step 6: Export model."""
    print("\n" + "=" * 60)
    print("Step 6: Model Export")
    print("=" * 60)

    # Export to ONNX (cross-platform)
    onnx_path = os.path.join(output_dir, "model.onnx")
    trainer.export(onnx_path, format="onnx", quantize=quantize)
    print(f"Exported ONNX: {onnx_path} (quantize={quantize})")

    # Export to TorchScript
    torchscript_path = os.path.join(output_dir, "model.pt")
    trainer.export(torchscript_path, format="torchscript")
    print(f"Exported TorchScript: {torchscript_path}")

    print("\nAdditional export options (uncomment as needed):")
    print("  # trainer.export('model.xml', format='openvino', quantize='int8')")
    print("  # trainer.export('model.engine', format='tensorrt', quantize='fp16')")

    print("\nExport format comparison:")
    print("  ONNX: Cross-platform, best compatibility")
    print("  TorchScript: PyTorch ecosystem")
    print("  OpenVINO: Best for Intel hardware")
    print("  TensorRT: Best for NVIDIA hardware")


def main():
    parser = argparse.ArgumentParser(
        description="iVIT-SDK AI Developer Training Example"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (ImageFolder format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Pre-trained model name (default: resnet50)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training (default: cuda:0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Disable backbone freezing (fine-tune entire model)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("iVIT-SDK AI Developer Training Example")
    print("=" * 60)
    print(f"SDK Version: {ivit.__version__}")

    # Check dataset exists
    if not os.path.exists(args.dataset):
        print(f"\nError: Dataset not found at {args.dataset}")
        print("\nPlease prepare your dataset in ImageFolder format:")
        print("  my_dataset/")
        print("    class_a/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class_b/")
        print("      image1.jpg")
        print("      image2.jpg")
        sys.exit(1)

    # Step 1: Prepare datasets
    train_dataset, val_dataset = prepare_datasets(args.dataset)

    # Step 2: Create trainer
    trainer = create_trainer(
        model_name=args.model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        freeze_backbone=not args.no_freeze,
    )

    # Step 3: Setup callbacks
    callbacks = setup_callbacks(args.output)

    # Step 4: Train
    history = train_model(trainer, callbacks)

    # Step 5: Evaluate
    metrics = evaluate_model(trainer)

    # Step 6: Export
    export_model(trainer, args.output)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {args.output}/best_model.pt")
    print(f"ONNX model saved to: {args.output}/model.onnx")
    print(f"\nNext steps:")
    print(f"  1. Load exported model: ivit.load('{args.output}/model.onnx')")
    print(f"  2. Run inference: results = model('image.jpg')")


if __name__ == "__main__":
    main()