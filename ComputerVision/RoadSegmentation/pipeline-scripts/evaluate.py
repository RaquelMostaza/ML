import argparse
from pathlib import Path
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow import shape, reshape
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Model
from pickle import load
from pickle import dump
import joblib
import random
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D
from tqdm import tqdm
import h5py
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from azureml.core import Run, Model
import mlflow
import mlflow.sklearn


# current run
run = Run.get_context()
ws = run.experiment.workspace


def parse_args():

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--prepared_X", type=str, help="Path to store prepared images")
    parser.add_argument("--prepared_y", type=str, help="Path to store prepared masks")
    parser.add_argument("--predictions", type=str, help="Path of predictions")
    parser.add_argument("--score_report", type=str, help="Path to score report")
    parser.add_argument('--deploy_flag', type=str, help='A deployment flag whether to deployment or no')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()


    args = parse_args()

    test_images = np.load((Path(args.prepared_X) / "X_test.npy"))
    test_masks = np.load((Path(args.prepared_y) / "y_test.npy"))

    # Load the saved model
    model = tf.keras.models.load_model((Path(args.model_input) / "model.h5"))

    # Evaluate the model on the test data
    y_pred = []
    for i in tqdm(range(test_images.shape[0])):
        image = np.expand_dims(test_images[i], axis=0)
        mask_pred = model.predict(image)[0]
        mask_pred = (mask_pred > 0.5).astype(np.uint8)
        y_pred.append(mask_pred)

    y_pred = np.array(y_pred)

    # Calculate F1 score
    # f1 = f1_score(test_masks.flatten(), y_pred.flatten(), average="binary")

    # print("F1 score: {:.4f}".format(f1))

    # Save the output test-data with feature columns
    output_data = test_images.copy()
    output_data["real_label"] : test_masks
    output_data["predicted_label"] : y_pred

    prediction_path = (Path(args.predictions) /'predictions.npy')
    np.save(prediction_path,output_data)
    # output_data.to_c((Path(args.predictions) / "predictions.csv"), index=False)

   
    # Predict the masks on the test data using the U-Net model
    predicted_masks = model.predict(test_images)

    # Define the Dice coefficient metric (assuming labels are binary 0/1)
    def dice_coefficient(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2. * intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.)
        return dice


    # Compute the overall Dice coefficient on the test data
    dice_scores = np.zeros(predicted_masks.shape[0])
    for i in range(predicted_masks.shape[0]):
        dice_scores[i] = dice_coefficient(test_masks[i], predicted_masks[i])
        mean_dice = np.mean(dice_scores)

    # Compute the Dice coefficient for each predicted mask
    dice_scores = []
    for i in range(predicted_masks.shape[0]):
        intersection = np.sum(np.logical_and(predicted_masks[i], test_masks[i]))
        union = np.sum(np.logical_or(predicted_masks[i], test_masks[i]))
        dice_score = (2. * intersection + 1e-9) / (union + 1e-9)
        dice_scores.append(dice_score)

    # Print the mean Dice coefficient
    print('Mean Dice coefficient:', dice_scores)

    (Path(args.score_report) / "score.txt").write_text(
        "Scored with the following model:\n{}".format(model)
    )
    with open((Path(args.score_report) / "score.txt"), "a") as f:
        f.write("Dice scores: %s\n" % dice_scores)

    mlflow.log_metric("Dice score", mean_dice)
        
    # -------------------- Promotion ------------------- #
    scores = {}
    predictions = {}
    test_masks_2d = test_masks.reshape(test_masks.shape[0], -1)
    predicted_masks_2d = predicted_masks.reshape(predicted_masks.shape[0], -1)
    score = r2_score(test_masks_2d, predicted_masks_2d) # current model
    for model_run in Model.list(ws):
        if model_run.name == args.model_name:
            model_path = Model.download(model_run, exist_ok=True)
            mdl = tf.keras.models.load_model((Path(args.model_input) / "model.h5"))
            predictions[model_run.id] = mdl.predict(test_images)
            predictions[model_run.id] = predictions[model_run.id].reshape(predictions[model_run.id].shape[0], -1)
            scores[model_run.id] = r2_score(test_masks_2d, predictions[model_run.id])

    print(scores)
    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print("Deploy flag: ",deploy_flag)

    with open((Path(args.deploy_flag) / "deploy_flag"), 'w') as f:
        f.write('%d' % int(deploy_flag))

    scores["current model"] = score
    perf_comparison_plot = pd.DataFrame(scores, index=["r2 score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(args.score_report) / "perf_comparison.png")

    mlflow.log_metric("deployment flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    
if __name__ == "__main__":
    main()  
