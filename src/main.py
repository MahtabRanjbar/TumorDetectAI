# Import system libs
import os
from time import perf_counter

# Import deep learning libraries
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Import custom modules
from config import Config
from evaluation import (display_confusion_matrix, evaluate_model,
                        plot_training_history, save_classification_report)
from preprocess import (create_gen, create_image_dataframe,
                        generate_data_paths_labels)
from model import create_model


def main():
    # Generate data paths and labels
    filepaths, labels = generate_data_paths_labels(Config.data_dir)

    # Split the data into train, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=0.2,
        stratify=labels,
        shuffle=True,
        random_state=Config.seed,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        stratify=train_labels,
        shuffle=True,
        random_state=Config.seed,
    )

    # Create dataframes for train, validation, and test sets
    train_df = create_image_dataframe(train_paths)
    val_df = create_image_dataframe(val_paths)
    test_df = create_image_dataframe(test_paths)

    # Create the generators
    (
        train_generator,
        test_generator,
        train_images,
        val_images,
        test_images,
    ) = create_gen(train_df, val_df, test_df)

    # Get the model
    model = create_model()

    # Start the timer
    start = perf_counter()

    # Define early stopping callback
    early_stopping = EarlyStopping(
        patience=3, monitor="val_loss", restore_best_weights=True
    )

    # Fit the model
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=Config.epochs,
        verbose=1,
        callbacks=[early_stopping],
    )

    # Calculate the duration
    duration = round(perf_counter() - start, 2)

    # Print the training duration
    print(f"Trained in {duration} sec")

    # Map the labels
    labels = train_images.class_indices
    labels = {v: k for k, v in labels.items()}
    y_test = [labels[k] for k in test_images.labels]

    # Evaluate the model on test images and save the metrics to a text file
    evaluation = evaluate_model(model, test_images, y_test, path=Config.evaluation_path)
    print("Evaluation metrics saved to evaluation_metrics.txt")

    pred = evaluation["predictions"]

    # Plot training history
    plot_training_history(history, save_path=Config.training_history_path)

    # Display confusion matrix
    display_confusion_matrix(y_test, pred, save_path=Config.confusion_matrix_save_path)

    # Save classification report
    save_classification_report(
        y_test, pred, save_path=Config.classification_report_path
    )

    # Create the model directory if it doesn't exist
    os.makedirs(Config.model_dir, exist_ok=True)

    # Save the best model
    save_path = os.path.join(Config.model_dir, "best_model.h5")
    model.save(save_path)
    print(f"Best model saved at: {save_path}")


if __name__ == 'main':
    main()