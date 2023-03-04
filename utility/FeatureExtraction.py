import numpy as np

import cv2
from utility import Utils

from cv import FaceMeshDetectionModule as faceMesh

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
L_NASAL_LABIAL_FURROW = 1
R_NASAL_LABIAL_FURROW = 2
L_CLOSING_EYE = 3
R_CLOSING_EYE = 4


def compute_gradient(face_masks, frame, frame_debug):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grd = []
    avg_face_gradients = [0] * FACE_DISTANCES_DIMENSIONS

    for mask in face_masks:
        if len(gray_frame[mask == 255]) != 0:
            sobel_x = cv2.Sobel(gray_frame[mask == 255], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_frame[mask == 255], cv2.CV_64F, 0, 1, ksize=3)
            sobel_x_abs = np.uint8(np.absolute(sobel_x))
            sobel_y_abs = np.uint8(np.absolute(sobel_y))
            sobel_xy_abs = cv2.bitwise_or(sobel_x_abs, sobel_y_abs)
            grd.append(sobel_xy_abs)
            frame_debug[mask == 255] = sobel_xy_abs
        else:
            return frame_debug, avg_face_gradients

    grd[NASAL_WRINKLES] = sum(grd[NASAL_WRINKLES]) / len(grd[NASAL_WRINKLES])
    grd[L_NASAL_LABIAL_FURROW] = sum(grd[L_NASAL_LABIAL_FURROW]) / len(grd[L_NASAL_LABIAL_FURROW])
    grd[R_NASAL_LABIAL_FURROW] = sum(grd[R_NASAL_LABIAL_FURROW]) / len(grd[R_NASAL_LABIAL_FURROW])
    grd[L_CLOSING_EYE] = sum(grd[L_CLOSING_EYE]) / len(grd[L_CLOSING_EYE])
    grd[R_CLOSING_EYE] = sum(grd[R_CLOSING_EYE]) / len(grd[R_CLOSING_EYE])

    cv2.putText(frame_debug, f'Avg. Gradients per frame', (380, 20), Utils.FONT, 0.5, Utils.YELLOW, 2)
    cv2.putText(frame_debug, f'Nasal wrinkles gradient: {grd[NASAL_WRINKLES][0]:.4f}',
                (380, 40), Utils.FONT, 0.4, Utils.YELLOW, 1)
    cv2.putText(frame_debug, f'Left nasal labial furrow gradient: {grd[L_NASAL_LABIAL_FURROW][0]:.4f}',
                (380, 60), Utils.FONT, 0.4, Utils.YELLOW, 1)
    cv2.putText(frame_debug, f'Right nasal labial furrow gradient: {grd[R_NASAL_LABIAL_FURROW][0]:.4f}',
                (380, 80), Utils.FONT, 0.4, Utils.YELLOW, 1)
    cv2.putText(frame_debug, f'Left closing eye gradient: {grd[L_CLOSING_EYE][0]:.4f}',
                (380, 100), Utils.FONT, 0.4, Utils.YELLOW, 1)
    cv2.putText(frame_debug, f'Right closing eye gradient: {grd[R_CLOSING_EYE][0]:.4f}',
                (380, 120), Utils.FONT, 0.4, Utils.YELLOW, 1)

    avg_face_gradients = grd[0].tolist() + grd[1].tolist() + grd[2].tolist() + grd[3].tolist() + grd[4].tolist()

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
    face_mesh_detector = faceMesh.FaceMeshDetector()
    toggle = True

    while True:
        is_true, frame = capture.read()
        frame = cv2.flip(frame, 1)
        cv2.namedWindow(file_name, cv2.WND_PROP_TOPMOST)
        cv2.setWindowProperty(file_name, cv2.WND_PROP_TOPMOST, cv2.WND_PROP_TOPMOST)
        if frame is None:
            break
        frame_debug_gradients = frame.copy()
        frame_debug_distances, head_position, distances, masks = face_mesh_detector.extract_frame_distances(frame)

        if len(head_position) != 0:
            hpe = add_hpe_contributes(head_position)
            hpe_frames += 1
        if len(masks) != 0:
            frame_debug_gradients, face_gradients = compute_gradient(masks, frame, frame_debug_distances.copy())
            gradient_frames += 1
        if len(distances) != 0:
            face_distances = add_face_distances_contributes(distances)
            distances_frames += 1

        key = cv2.waitKey(5)
        if key == 9:
            toggle = not toggle
        elif key == 27:
            exit(1)

        if toggle:
            cv2.imshow(file_name, frame_debug_distances)
        else:
            cv2.imshow(file_name, frame_debug_gradients)
    cv2.destroyAllWindows()

    if hpe_frames > 0:
        avg_hpe = [float(x / hpe_frames) for x in hpe]
    if distances_frames > 0:
        avg_video_distances = [float(x / distances_frames) for x in face_distances]
    if gradient_frames > 0:
        avg_face_gradients = [float(x / gradient_frames) for x in face_gradients]

    return avg_hpe, avg_face_gradients, avg_video_distances


def extract_video_features():
    capture = cv2.VideoCapture(0)
    avg_video_features(capture, "Camera")
    capture.release()
