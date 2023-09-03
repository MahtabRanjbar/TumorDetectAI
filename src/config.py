class Config:
    # Paths
    #"/kaggle/input/brain-mri-images-for-brain-tumor-detection
    data_dir = 'data'
    evaluation_path = "evaluation_metrics.txt"
    training_history_path = "training_history.png"
    confusion_matrix_save_path = "/kaggle/working/confusion_matrix.png"
    classification_report_path = "/kaggle/working/classification_report.txt"
    model_dir = "saved_models"

    # Model training configuration
    seed = 42
    epochs = 15
    input_size = (224, 224)
    weights = "imagenet"
    learning_rate = 0.001

    # Generator configuration
    generator_args = {
        "class_mode": "categorical",
        "batch_size": 8,
        "seed": seed,
        "target_size": input_size,
        "color_mode": "rgb",
    }

    # Augmentation configuration
    augmentation_args = {
        "rotation_range": 30,
        "zoom_range": 0.15,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.15,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    }
