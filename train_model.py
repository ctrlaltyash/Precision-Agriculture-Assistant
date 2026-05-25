import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_IMG_HEIGHT = 224
DEFAULT_IMG_WIDTH = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50


def validate_directories(train_dir: Path, valid_dir: Path) -> bool:
    """
    Validate that training and validation directories exist and contain subdirectories.
    
    Args:
        train_dir: Path to training directory
        valid_dir: Path to validation directory
        
    Returns:
        bool: True if both directories are valid
    """
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        return False
    if not valid_dir.exists():
        logger.error(f"Validation directory not found: {valid_dir}")
        return False
    
    # Check for subdirectories (class folders)
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    valid_classes = [d for d in valid_dir.iterdir() if d.is_dir()]
    
    if not train_classes:
        logger.error(f"No class directories found in {train_dir}")
        return False
    
    logger.info(f"Found {len(train_classes)} classes in training directory")
    logger.info(f"Found {len(valid_classes)} classes in validation directory")
    
    return True


def build_model(num_classes: int, img_height: int, img_width: int) -> Sequential:
    """
    Build a CNN model for plant disease classification.
    
    Args:
        num_classes: Number of disease classes
        img_height: Image height
        img_width: Image width
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train(
    plant_type: str,
    base_train_dir: str = 'data/train',
    base_valid_dir: str = 'data/valid',
    model_output: Optional[str] = None,
    indices_output: Optional[str] = None,
    img_height: int = DEFAULT_IMG_HEIGHT,
    img_width: int = DEFAULT_IMG_WIDTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS
) -> bool:
    """
    Train a plant disease classification model.
    
    Args:
        plant_type: Plant type to train on (e.g., 'Tomato', 'Apple')
        base_train_dir: Base directory for training data
        base_valid_dir: Base directory for validation data
        model_output: Path to save model (default: {plant_type}_model.h5)
        indices_output: Path to save class indices (default: {plant_type}_indices.json)
        img_height: Image height in pixels
        img_width: Image width in pixels
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        
    Returns:
        bool: True if training succeeded
    """
    # Set up paths
    train_dir = Path(base_train_dir) / plant_type
    valid_dir = Path(base_valid_dir) / plant_type
    model_path = model_output or f'{plant_type}_model.h5'
    indices_path = indices_output or f'{plant_type}_indices.json'
    
    # Validate directories
    logger.info(f"Validating directories for {plant_type}...")
    if not validate_directories(train_dir, valid_dir):
        return False
    
    # Data Augmentation for training data
    logger.info("Setting up data generators with augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation data, only rescaling
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        logger.info(f"Loading training data from {train_dir}...")
        train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        logger.info(f"Loading validation data from {valid_dir}...")
        validation_generator = valid_datagen.flow_from_directory(
            str(valid_dir),
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Save class indices for later inference
    logger.info("Saving class indices...")
    class_indices = train_generator.class_indices
    indices_to_class = {v: k for k, v in class_indices.items()}
    try:
        with open(indices_path, 'w') as f:
            json.dump(indices_to_class, f, indent=2)
        logger.info(f"Class indices saved to {indices_path}")
    except Exception as e:
        logger.error(f"Failed to save class indices: {e}")
        return False
    
    # Build model
    logger.info("Building model architecture...")
    model = build_model(
        num_classes=train_generator.num_classes,
        img_height=img_height,
        img_width=img_width
    )
    model.summary()
    
    # Set up callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    logger.info("Starting training...")
    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            workers=1,
            use_multiprocessing=False
        )
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False
    
    # Final model save
    model.save(model_path)
    logger.info(f"Final model saved to {model_path}")
    
    return True


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train a plant disease classification model'
    )
    parser.add_argument(
        'plant_type',
        help='Plant type to train on (e.g., Tomato, Apple)'
    )
    parser.add_argument(
        '--train-dir',
        default='data/train',
        help='Base directory for training data (default: data/train)'
    )
    parser.add_argument(
        '--valid-dir',
        default='data/valid',
        help='Base directory for validation data (default: data/valid)'
    )
    parser.add_argument(
        '--model-output',
        help='Path to save model (default: {plant_type}_model.h5)'
    )
    parser.add_argument(
        '--indices-output',
        help='Path to save class indices (default: {plant_type}_indices.json)'
    )
    parser.add_argument(
        '--img-height',
        type=int,
        default=DEFAULT_IMG_HEIGHT,
        help=f'Image height (default: {DEFAULT_IMG_HEIGHT})'
    )
    parser.add_argument(
        '--img-width',
        type=int,
        default=DEFAULT_IMG_WIDTH,
        help=f'Image width (default: {DEFAULT_IMG_WIDTH})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Maximum epochs (default: {DEFAULT_EPOCHS})'
    )
    
    args = parser.parse_args()
    
    success = train(
        plant_type=args.plant_type,
        base_train_dir=args.train_dir,
        base_valid_dir=args.valid_dir,
        model_output=args.model_output,
        indices_output=args.indices_output,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
