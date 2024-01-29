import os
import pandas as pd

from utility import DataGeneration

from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

BIOSIGNALS_BINARY_PATH = "features/extracted_bio_features_binary.csv"
VIDEO_FEATURES_BINARY_PATH = "features/extracted_video_features_binary.csv"
EARLY_FUSED_FEATURES_BINARY_PATH = "features/fused_features_binary.csv"

BIOSIGNALS_MULTI_PATH = "features/extracted_bio_features.csv"
VIDEO_FEATURES_MULTI_PATH = "features/extracted_video_features.csv"
EARLY_FUSED_FEATURES_MULTI_PATH = "features/fused_features.csv"


def evaluate_accuracy(X, y, support_vector_machine):
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
    clf = support_vector_machine.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Accuracy:", accuracy)
    return clf


def train_svm_binary_biosignals():
    svm_biosignals_binary = svm.SVC(kernel="linear", C=1)
    X, y = DataGeneration.io_sets_binary(BIOSIGNALS_BINARY_PATH)
    print("SVM Biosignals Binary Classification")
    svm_biosignals_binary = evaluate_accuracy(X, y, svm_biosignals_binary)
    return svm_biosignals_binary


def train_svm_binary_video():
    svm_video_binary = svm.SVC(kernel="poly", C=1)
    X, y = DataGeneration.io_sets_binary(VIDEO_FEATURES_BINARY_PATH)
    # X = X.drop(['NOSE_X', 'NOSE_Y', 'NOSE_Z'], axis=1)
    # X = X.drop(['HEAD_ROTATION_X', 'HEAD_ROTATION_Y', 'HEAD_ROTATION_Z'], axis=1)
    X = X.drop(['CORRUGATOR_GRAD'], axis=1)
    print("SVM Video Binary Classification")
    svm_video_binary = evaluate_accuracy(X, y, svm_video_binary)
    return svm_video_binary


def train_svm_early_fusion():
    svm_early_fusion = svm.SVC(kernel="linear", C=1)
    X, y = DataGeneration.io_sets_binary(EARLY_FUSED_FEATURES_BINARY_PATH)
    X = X.drop(['NOSE_X', 'NOSE_Y', 'NOSE_Z'], axis=1)
    X = X.drop(['CORRUGATOR_GRAD'], axis=1)
    print("SVM Early Fusion Binary Classification")
    svm_early_fusion = evaluate_accuracy(X, y, svm_early_fusion)
    return svm_early_fusion


def train_svm_ecg():
    svm_ecg = svm.SVC(kernel="rbf", C=1)
    X, y = DataGeneration.io_sets_binary(BIOSIGNALS_BINARY_PATH)
    X = X.drop(['MEAN_GSR', 'MEDIAN_GSR', 'MAX_GSR', 'MIN_GSR', 'STD_GSR', 'SUM_GSR', 'VAR_GSR',
                'MEAN_EMG_TRAPEZIUS', 'MEDIAN_EMG_TRAPEZIUS', 'MAX_EMG_TRAPEZIUS', 'MIN_EMG_TRAPEZIUS',
                'STD_EMG_TRAPEZIUS', 'SUM_EMG_TRAPEZIUS', 'VAR_EMG_TRAPEZIUS'], axis=1)
    svm_ecg = evaluate_accuracy(X, y, svm_ecg)
    return svm_ecg


def train_svm_gsr():
    svm_gsr = svm.SVC(kernel="rbf", C=1)
    X, y = DataGeneration.io_sets_binary(BIOSIGNALS_BINARY_PATH)
    X = X.drop(['MEAN_ECG', 'MEDIAN_ECG', 'MAX_ECG', 'MIN_ECG', 'STD_ECG', 'SUM_ECG', 'VAR_ECG',
                'MEAN_EMG_TRAPEZIUS', 'MEDIAN_EMG_TRAPEZIUS', 'MAX_EMG_TRAPEZIUS', 'MIN_EMG_TRAPEZIUS',
                'STD_EMG_TRAPEZIUS', 'SUM_EMG_TRAPEZIUS', 'VAR_EMG_TRAPEZIUS'], axis=1)
    svm_gsr = evaluate_accuracy(X, y, svm_gsr)
    return svm_gsr


def train_svm_emg_trapezius():
    svm_emg_trapezius = svm.SVC(kernel="rbf", C=1)
    X, y = DataGeneration.io_sets_binary(BIOSIGNALS_BINARY_PATH)
    X = X.drop(['MEAN_ECG', 'MEDIAN_ECG', 'MAX_ECG', 'MIN_ECG', 'STD_ECG', 'SUM_ECG', 'VAR_ECG',
                'MEAN_GSR', 'MEDIAN_GSR', 'MAX_GSR', 'MIN_GSR', 'STD_GSR', 'SUM_GSR', 'VAR_GSR'], axis=1)
    svm_emg_trapezius = evaluate_accuracy(X, y, svm_emg_trapezius)
    return svm_emg_trapezius


def late_fusion():
    print("Late fusion (training the SVMs...)")
    X_ecg, y_ecg = DataGeneration.io_sets_ecg_only()
    x_train_ecg, x_test_ecg, y_train_ecg, y_test_ecg = train_test_split(X_ecg, y_ecg, random_state=0, test_size=0.25)
    svm_ecg = svm.SVC(kernel="linear", C=1).fit(x_train_ecg, y_train_ecg)
    print("ECG accuracy:")
    evaluate_accuracy(X_ecg, y_ecg, svm_ecg)

    X_gsr, y_gsr = DataGeneration.io_sets_gsr_only()
    x_train_gsr, x_test_gsr, y_train_gsr, y_test_gsr = train_test_split(X_gsr, y_gsr, random_state=0, test_size=0.25)
    svm_gsr = svm.SVC(kernel="linear", C=1).fit(x_train_gsr, y_train_gsr)
    print("GSR accuracy:")
    evaluate_accuracy(X_gsr, y_gsr, svm_gsr)

    X_video, y_video = DataGeneration.io_sets_video_only()
    x_train_video, x_test_video, y_train_video, y_test_video = train_test_split(X_video, y_video, random_state=0,
                                                                                test_size=0.25)
    svm_video = svm.SVC(kernel="linear", C=1).fit(x_train_video, y_train_video)
    print("Video accuracy:")
    evaluate_accuracy(X_video, y_video, svm_video)

    X, y = DataGeneration.io_sets_binary(EARLY_FUSED_FEATURES_BINARY_PATH)
    # X = X.drop(['HEAD_ROTATION_X', 'HEAD_ROTATION_Y', 'HEAD_ROTATION_Z'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    correct_pred = 0
    for index in range(len(x_test)):
        ecg_pred = svm_ecg.predict([x_test.iloc[index, 0:7]])
        gsr_pred = svm_gsr.predict([x_test.iloc[index, 7:14]])
        video_pred = svm_video.predict([x_test.iloc[index, 21:]])

        global_pred = ecg_pred + gsr_pred + video_pred
        if global_pred > 1:
            if y_test.iloc[index] == 1:
                correct_pred = correct_pred + 1
        else:
            if y_test.iloc[index] == 0:
                correct_pred = correct_pred + 1

    accuracy = correct_pred * 100 / len(x_test)
    print(f"Late fusion accuracy: {accuracy}")


def binary_classification():
    svm_biosignals_binary = train_svm_binary_biosignals()
    svm_video_binary = train_svm_binary_video()
    svm_early_fusion = train_svm_early_fusion()

    train_svm_ecg()
    train_svm_gsr()
    train_svm_emg_trapezius()
    train_svm_binary_video()

    late_fusion()


def multiclass_classification():
    svm_biosignals_binary = svm.SVC(kernel=" ", C=1)
    X, y = DataGeneration.io_sets_multiclass(BIOSIGNALS_MULTI_PATH)
    print("SVM Biosignals Multiclass Classification")
    svm_biosignals_binary = evaluate_accuracy(X, y, svm_biosignals_binary)

    svm_video_binary = svm.SVC(kernel=KERNEL, C=1)
    X, y = DataGeneration.io_sets_multiclass(VIDEO_FEATURES_MULTI_PATH)
    X = X.drop(['NOSE_X', 'NOSE_Y', 'NOSE_Z'], axis=1)
    # X = X.drop(['HEAD_ROTATION_X', 'HEAD_ROTATION_Y', 'HEAD_ROTATION_Z'], axis=1)
    X = X.drop(['CORRUGATOR_GRAD'], axis=1)
    print("SVM Video Multiclass Classification")
    svm_video_binary = evaluate_accuracy(X, y, svm_video_binary)

    svm_early_fusion = svm.SVC(kernel=KERNEL, C=1)
    X, y = DataGeneration.io_sets_multiclass(EARLY_FUSED_FEATURES_MULTI_PATH)
    X = X.drop(['NOSE_X', 'NOSE_Y', 'NOSE_Z'], axis=1)
    X = X.drop(['CORRUGATOR_GRAD'], axis=1)
    print("SVM Early Fusion Multiclass Classification")
    svm_early_fusion = evaluate_accuracy(X, y, svm_early_fusion)

    # Late fusion approach
    print("Late fusion approach")
    svm_ecg = svm.SVC(kernel=KERNEL, C=1)
    X, y = DataGeneration.io_sets_multiclass(BIOSIGNALS_MULTI_PATH)
    X = X.drop(['MEAN_GSR', 'MEDIAN_GSR', 'MAX_GSR', 'MIN_GSR', 'STD_GSR', 'SUM_GSR', 'VAR_GSR',
                'MEAN_EMG_TRAPEZIUS', 'MEDIAN_EMG_TRAPEZIUS', 'MAX_EMG_TRAPEZIUS', 'MIN_EMG_TRAPEZIUS',
                'STD_EMG_TRAPEZIUS', 'SUM_EMG_TRAPEZIUS', 'VAR_EMG_TRAPEZIUS'], axis=1)
    print("SVM ECG Multiclass Classification")
    svm_ecg = evaluate_accuracy(X, y, svm_ecg)

    svm_gsr = svm.SVC(kernel=KERNEL, C=1)
    X, y = DataGeneration.io_sets_multiclass(BIOSIGNALS_MULTI_PATH)
    X = X.drop(['MEAN_ECG', 'MEDIAN_ECG', 'MAX_ECG', 'MIN_ECG', 'STD_ECG', 'SUM_ECG', 'VAR_ECG',
                'MEAN_EMG_TRAPEZIUS', 'MEDIAN_EMG_TRAPEZIUS', 'MAX_EMG_TRAPEZIUS', 'MIN_EMG_TRAPEZIUS',
                'STD_EMG_TRAPEZIUS', 'SUM_EMG_TRAPEZIUS', 'VAR_EMG_TRAPEZIUS'], axis=1)
    print("SVM GSR Multiclass Classification")
    svm_gsr = evaluate_accuracy(X, y, svm_gsr)

    svm_emg_trapezius = svm.SVC(kernel=KERNEL, C=1)
    X, y = DataGeneration.io_sets_binary(BIOSIGNALS_MULTI_PATH)
    X = X.drop(['MEAN_ECG', 'MEDIAN_ECG', 'MAX_ECG', 'MIN_ECG', 'STD_ECG', 'SUM_ECG', 'VAR_ECG',
                'MEAN_GSR', 'MEDIAN_GSR', 'MAX_GSR', 'MIN_GSR', 'STD_GSR', 'SUM_GSR', 'VAR_GSR'], axis=1)
    print("SVM EMG Trapezius Multiclass Classification")
    svm_emg_trapezius = evaluate_accuracy(X, y, svm_emg_trapezius)
