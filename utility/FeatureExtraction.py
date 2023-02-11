import numpy
import pandas as pd
import numpy as np
import math
import os
import cv2
import csv

from cv import FaceMeshDetectionModule as face_mesh_module

CSV_HEADER_VIDEO_FEATURES = ['SUBJECT_ID', 'TRIAL', 'AGE', 'SEX', 'PAIN_LEVEL',
                             'NOSE_X', 'NOSE_Y', 'NOSE_Z',
                             'HEAD_ROTATION_X', 'HEAD_ROTATION_Y', 'HEAD_ROTATION_Z',
                             'MOUTH_HORIZONTAL', 'MOUTH_VERTICAL',
                             'LEFT_EYEBROW_IRIS', 'LEFT_IRIS_MOUTH', 'LEFT_EYEBROW_MOUTH',
                             'RIGHT_EYEBROW_IRIS', 'RIGHT_IRIS_MOUTH', 'RIGHT_EYEBROW_MOUTH',
                             'CORRUGATOR_GRAD', 'LEFT_SMILE_GRAD', 'RIGHT_SMILE_GRAD', 'LEFT_EYE_GRAD', 'RIGHT_EYE_GRAD'
                             ]

CSV_HEADER_BIO_FEATURES = ['SUBJECT_ID', 'TRIAL', 'PAIN_LEVEL', 'AGE', 'SEX',
                           'MEAN_GSR', 'MEDIAN_GSR', 'MAX_GSR', 'MIN_GSR', 'STD_GSR', 'SUM_GSR', 'VAR_GSR',
                           'MEAN_ECG', 'MEDIAN_ECG', 'MAX_ECG', 'MIN_ECG', 'STD_ECG', 'SUM_ECG', 'VAR_ECG',
                           'MEAN_EMG_TRAPEZIUS', 'MEDIAN_EMG_TRAPEZIUS', 'MAX_EMG_TRAPEZIUS', 'MIN_EMG_TRAPEZIUS',
                           'STD_EMG_TRAPEZIUS', 'SUM_EMG_TRAPEZIUS', 'VAR_EMG_TRAPEZIUS',
                           'MEAN_EMG_CORRUGATOR', 'MEDIAN_EMG_CORRUGATOR', 'MAX_EMG_CORRUGATOR', 'MIN_EMG_CORRUGATOR',
                           'STD_EMG_CORRUGATOR', 'SUM_EMG_CORRUGATOR', 'VAR_EMG_CORRUGATOR',
                           'MEAN_EMG_ZYGOMATICUS', 'MEDIAN_EMG_ZYGOMATICUS', 'MAX_EMG_ZYGOMATICUS',
                           'MIN_EMG_ZYGOMATICUS',
                           'STD_EMG_ZYGOMATICUS', 'SUM_EMG_ZYGOMATICUS', 'VAR_EMG_ZYGOMATICUS']
# Head position estimation features (hpe)
HPE_DIMENSIONS = 6
NOSE_X = 0
NOSE_Y = 1
NOSE_Z = 2
HEAD_ROT_X = 3
HEAD_ROT_Y = 4
HEAD_ROT_Z = 5

# Distance features
FACE_DISTANCES_DIMENSIONS = 9
MOUTH_H = 0
MOUTH_V = 1
L_EYEBROW_IRIS = 2
L_IRIS_MOUTH = 3
L_EYEBROW_MOUTH = 4
R_EYEBROW_IRIS = 5
R_IRIS_MOUTH = 6
R_EYEBROW_MOUTH = 7
EYE_AXIS = 8

# Gradient features
FACE_GRADIENTS_DIMENSIONS = 5
NASAL_WRINKLES = 0
L_NASOLABIAL_FURROW = 1
R_NASOLABIAL_FURROW = 2
L_CLOSING_EYE = 3
R_CLOSING_EYE = 4


def rescaleFrame(image, scale=0.75):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def statistical_indices_column(df, column):
    return df[column].mean(), df[column].median(), \
        df[column].max(), df[column].min(), \
        df[column].std(), df[column].sum(), \
        df[column].var()


def extract_biosignals_features():
    f = open('features/extracted_bio_features.csv', 'w', encoding='UTF-8', newline='')
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(CSV_HEADER_BIO_FEATURES)
    bio_path = "D:\\BioVid_pain\\PartA\\biosignals_filtered"

    for root, dirs, files in os.walk(bio_path):
        for file_name in files:
            if file_name[-3:] == 'csv':
                df = pd.read_csv(os.path.join(root, file_name), sep='\t')
                df = df.iloc[1:, :]  # drop the first row
                mean_gsr, median_gsr, max_gsr, min_gsr, std_gsr, sum_gsr, var_gsr = \
                    statistical_indices_column(df, 'gsr')
                mean_ecg, median_ecg, max_ecg, min_ecg, std_ecg, sum_ecg, var_ecg = \
                    statistical_indices_column(df, 'ecg')

                mean_emg_trapezius, median_emg_trapezius, \
                    max_emg_trapezius, min_emg_trapezius, std_emg_trapezius, \
                    sum_emg_trapezius, var_emg_trapezius = statistical_indices_column(df, 'emg_trapezius')

                mean_emg_corrugator, median_emg_corrugator, \
                    max_emg_corrugator, min_emg_corrugator, std_emg_corrugator, \
                    sum_emg_corrugator, var_emg_corrugator = statistical_indices_column(df, 'emg_corrugator')

                mean_emg_zygomaticus, median_emg_zygomaticus, \
                    max_emg_zygomaticus, min_emg_zygomaticus, std_emg_zygomaticus, \
                    sum_emg_zygomaticus, var_emg_zygomaticus = statistical_indices_column(df, 'emg_zygomaticus')

                patient_info, pain_level, trial = file_name[0:-4].split('-')
                id, sex, age = patient_info.split('_')

                row = [id, trial, age, sex, pain_level,
                       mean_gsr, median_gsr, max_gsr, min_gsr, sum_gsr, std_gsr, var_gsr,
                       mean_ecg, median_ecg, max_ecg, min_ecg, sum_ecg, std_ecg, var_ecg,
                       mean_emg_trapezius, median_emg_trapezius, max_emg_trapezius, min_emg_trapezius,
                       sum_emg_trapezius, std_emg_trapezius, var_emg_trapezius,
                       mean_emg_corrugator, median_emg_corrugator, max_emg_corrugator, min_emg_corrugator,
                       sum_emg_corrugator, std_emg_corrugator, var_emg_corrugator,
                       mean_emg_zygomaticus, median_emg_zygomaticus, max_emg_zygomaticus, min_emg_zygomaticus,
                       sum_emg_zygomaticus, std_emg_zygomaticus, var_emg_zygomaticus]
                writer.writerow(row)

    f.close()


def compute_gradient(face_masks, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_debug = frame.copy()
    gradients = []
    avg_face_gradients = [0] * FACE_DISTANCES_DIMENSIONS

    for mask in face_masks:
        if len(gray_frame[mask == 255]) != 0:
            sobel_x = cv2.Sobel(gray_frame[mask == 255], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_frame[mask == 255], cv2.CV_64F, 0, 1, ksize=3)
            sobel_X_abs = np.uint8(np.absolute(sobel_x))
            sobel_Y_abs = np.uint8(np.absolute(sobel_y))
            sobel_XY_abs = cv2.bitwise_or(sobel_X_abs, sobel_Y_abs)
            gradients.append(sobel_XY_abs)
            frame_debug[mask == 255] = sobel_XY_abs
        else:
            return frame_debug, avg_face_gradients

    gradients[NASAL_WRINKLES] = sum(gradients[NASAL_WRINKLES]) / len(gradients[NASAL_WRINKLES])
    gradients[L_NASOLABIAL_FURROW] = sum(gradients[L_NASOLABIAL_FURROW]) / len(gradients[L_NASOLABIAL_FURROW])
    gradients[R_NASOLABIAL_FURROW] = sum(gradients[R_NASOLABIAL_FURROW]) / len(gradients[R_NASOLABIAL_FURROW])
    gradients[L_CLOSING_EYE] = sum(gradients[L_CLOSING_EYE]) / len(gradients[L_CLOSING_EYE])
    gradients[R_CLOSING_EYE] = sum(gradients[R_CLOSING_EYE]) / len(gradients[R_CLOSING_EYE])

    avg_face_gradients = gradients[0].tolist() + gradients[1].tolist() + \
                         gradients[2].tolist() + gradients[3].tolist() + \
                         gradients[4].tolist()

    return frame_debug, avg_face_gradients


def add_hpe_contributes(head_position):
    hpe = [0] * HPE_DIMENSIONS
    hpe[NOSE_X] += head_position[NOSE_X]
    hpe[NOSE_Y] += head_position[NOSE_Y]
    hpe[NOSE_Z] += head_position[NOSE_Z]
    hpe[HEAD_ROT_X] += head_position[HEAD_ROT_X]
    hpe[HEAD_ROT_Y] += head_position[HEAD_ROT_Y]
    hpe[HEAD_ROT_Z] += head_position[HEAD_ROT_Z]
    return hpe


def add_face_distances_contributes(distances):
    face_distances = [0] * FACE_DISTANCES_DIMENSIONS
    # eye_axis_factor = distances[EYE_AXIS]

    face_distances[MOUTH_H] += distances[MOUTH_H]
    face_distances[MOUTH_V] += distances[MOUTH_V]
    face_distances[L_EYEBROW_IRIS] += distances[L_EYEBROW_IRIS]
    face_distances[L_IRIS_MOUTH] += distances[L_IRIS_MOUTH]
    face_distances[L_EYEBROW_MOUTH] += distances[L_EYEBROW_MOUTH]
    face_distances[R_EYEBROW_IRIS] += distances[R_EYEBROW_IRIS]
    face_distances[R_IRIS_MOUTH] += distances[R_IRIS_MOUTH]
    face_distances[R_EYEBROW_MOUTH] += distances[R_EYEBROW_MOUTH]

    return face_distances


def avg_video_features(capture, file_name):
    hpe, face_distances, face_gradients = [0] * 6, [0] * 8, [0] * 5
    distances_frames, gradient_frames, hpe_frames = 0, 0, 0
    avg_hpe, avg_video_distances, avg_face_gradients = hpe, face_distances, face_gradients
    face_mesh_detector = face_mesh_module.FaceMeshDetector()

    while True:
        isTrue, frame = capture.read()
        if frame is None:
            break

        frame_debug_distances, head_position, distances, masks = face_mesh_detector.extract_frame_features(frame)

        if len(head_position) != 0:
            hpe = add_hpe_contributes(head_position)
            hpe_frames += 1

        if len(masks) != 0:
            frame_debug_gradient, face_gradients = compute_gradient(masks, frame)
            gradient_frames += 1

        if len(distances) != 0:
            face_distances = add_face_distances_contributes(distances)
            distances_frames += 1

        cv2.imshow(file_name, frame_debug_distances)
        cv2.moveWindow(file_name, 0, 0)

        if cv2.waitKey(5) & 0xFF == 27:
            continue

    cv2.destroyAllWindows()

    if hpe_frames > 0:
        avg_hpe = [float(x / hpe_frames) for x in hpe]
    if distances_frames > 0:
        avg_video_distances = [float(x / distances_frames) for x in face_distances]
    if gradient_frames > 0:
        avg_face_gradients = [float(x / gradient_frames) for x in face_gradients]

    return avg_hpe, avg_face_gradients, avg_video_distances


def extract_video_features():
    video_path = "D:\\BioVid_pain\\PartA\\video"

    f = open('features/extracted_video_features.csv', 'w', encoding='UTF-8', newline='')
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(CSV_HEADER_VIDEO_FEATURES)

    for root, dirs, files in os.walk(video_path):
        for file_name in files:
            if file_name[-3:] == 'mp4':
                capture = cv2.VideoCapture(os.path.join(root, file_name))
                avg_hpe, avg_face_gradients, avg_video_distances = avg_video_features(capture, file_name)
                capture.release()

                patient_info, pain_level, trial = file_name[0:-4].split('-')
                id, sex, age = patient_info.split('_')

                row = [id, trial, age, sex, pain_level,
                       avg_hpe[NOSE_X],
                       avg_hpe[NOSE_Y],
                       avg_hpe[NOSE_Z],
                       avg_hpe[HEAD_ROT_X],
                       avg_hpe[HEAD_ROT_Y],
                       avg_hpe[HEAD_ROT_Z],
                       avg_video_distances[MOUTH_H],
                       avg_video_distances[MOUTH_V],
                       avg_video_distances[L_EYEBROW_IRIS],
                       avg_video_distances[L_IRIS_MOUTH],
                       avg_video_distances[L_EYEBROW_MOUTH],
                       avg_video_distances[R_EYEBROW_IRIS],
                       avg_video_distances[R_IRIS_MOUTH],
                       avg_video_distances[R_EYEBROW_MOUTH],
                       avg_face_gradients[NASAL_WRINKLES],
                       avg_face_gradients[L_NASOLABIAL_FURROW],
                       avg_face_gradients[R_NASOLABIAL_FURROW],
                       avg_face_gradients[L_CLOSING_EYE],
                       avg_face_gradients[R_CLOSING_EYE]]
                writer.writerow(row)

            print(f"{file_name} video features written on extracted_video_features.csv")
    f.close()
    print("All video features got extracted successfully and written on 'extracted_video_features.csv'")
