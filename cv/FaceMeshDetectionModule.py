import mediapipe as mp
import mediapipe.framework.formats.landmark_pb2
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2
import math
import utility

# Colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Landmarks for distance features
NOSE_LANDMARK = 1
UPPER_CENTRAL_LIP_LANDMARK = 0
LOWER_CENTRAL_LIP_LANDMARK = 17
RIGHT_EYEBROW_LANDMARK = 282
LEFT_EYEBROW_LANDMARK = 52
LEFT_MOUTH_LANDMARK = 61
RIGHT_MOUTH_LANDMARK = 291

# Landmarks for gradient features
NASAL_WRINKLES_0 = 108
NASAL_WRINKLES_1 = 357

L_SMILE_FOLD_0 = 101
L_SMILE_FOLD_1 = 49
L_SMILE_FOLD_2 = 61
L_SMILE_FOLD_3 = 214
R_SMILE_FOLD_0 = 279
R_SMILE_FOLD_1 = 330
R_SMILE_FOLD_2 = 434
R_SMILE_FOLD_3 = 291

L_EYE_BBOX_0 = 225
L_EYE_BBOX_1 = 221
L_EYE_BBOX_2 = 233
L_EYE_BBOX_3 = 228

R_EYE_BBOX_0 = 445
R_EYE_BBOX_1 = 441
R_EYE_BBOX_2 = 453
R_EYE_BBOX_3 = 448


def rotate(points, angle):
    ANGLE = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
            ]
            for px, py in points
        ]
    ).astype(int)


def euclidean_distance(p0, p1):
    p0p1x = p0[0] - p1[0]
    p0p1y = p0[1] - p1[1]
    return math.sqrt(abs(p0p1x ^ 2 - p0p1y ^ 2))


def looking_direction(x, y):
    text = "Forward"
    if y < -10:
        text = "Looking Left"
    elif y > 10:
        text = "Looking Right"
    elif x < -10:
        text = "Looking Down"
    elif x > 10:
        text = "Looking Up"
    return text


def frame_text_direction_and_angles(frame_debug, direction_text, width, x, y, z):
    cv2.putText(frame_debug, direction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    str_angle_x = str(np.round(x, 2))
    str_angle_y = str(np.round(y, 2))
    str_angle_z = str(np.round(z, 2))
    cv2.putText(frame_debug, "x: " + str_angle_x, (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    cv2.putText(frame_debug, "y: " + str_angle_y, (width - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    cv2.putText(frame_debug, "z: " + str_angle_z, (width - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    return frame_debug


def distance_between_two_lm(frame, lm0, lm1, color):
    p0 = (int(lm0[0]), int(lm0[1]))
    p1 = (int(lm1[0]), int(lm1[1]))
    distance = euclidean_distance(p0, p1)
    cv2.line(frame, p0, p1, color, 1)
    return frame, distance


def get_rotation_angles(face_2d, face_3d, img_w, img_h):
    # Convert it to the NumPy array
    np_face_2d = np.array(face_2d, dtype=np.float64)
    # Convert it to the NumPy array
    np_face_3d = np.array(face_3d, dtype=np.float64)
    # Compute camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    # Compute distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(np_face_3d, np_face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rot_matrix, jac = cv2.Rodrigues(rot_vec)
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)
    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    return x, y, z


class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=1, refine_landmarks=True, min_detection_con=0.5, min_track_con=0.5):
        self.results = None
        self.staticMode = static_mode
        self.maxFaces = max_faces
        self.refineLandmarks = refine_landmarks
        self.minDetectionCon = min_detection_con
        self.minTrackCon = min_track_con
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLandmarks,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.connDrawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=GREEN)

        # 2D Face landmarks
        self.nose_2d = None
        self.nose_3d = None
        self.l_eyebrow = None
        self.l_mouth = None
        self.l_iris = None
        self.r_eyebrow = None
        self.r_mouth = None
        self.r_iris = None
        self.upper_lip = None
        self.lower_lip = None
        self.landmarks = None
        self.corrugator0 = None
        self.corrugator1 = None
        self.r_smile_fold0 = None
        self.r_smile_fold1 = None
        self.r_smile_fold2 = None
        self.r_smile_fold3 = None
        self.l_smile_fold0 = None
        self.l_smile_fold1 = None
        self.l_smile_fold2 = None
        self.l_smile_fold3 = None
        self.l_eye_bbox0 = None
        self.l_eye_bbox1 = None
        self.l_eye_bbox2 = None
        self.l_eye_bbox3 = None
        self.r_eye_bbox0 = None
        self.r_eye_bbox1 = None
        self.r_eye_bbox2 = None
        self.r_eye_bbox3 = None

        # Features vectors (total 19 features)
        self.head_position = [0] * 6  # 6D (pos, angles)
        self.facial_distances = []  # 8D
        self.facial_masks = []  # 5D

    def findNumberedFaceMesh(self, img, draw=True):
        self.results = self.faceMesh.process(img)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms,
                                           self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                           self.drawSpec, self.drawSpec)
                face_3d = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = lm.x * iw, lm.y * ih
                    int_x, int_y = int(x), int(y)
                    cv2.putText(img, str(id), (int_x, int_y), cv2.FONT_HERSHEY_PLAIN, 0.5, GREEN, 1)
                    face_3d.append([int_x, int_y, lm.z])
                faces.append(face_3d)
        return img, faces

    def get_iris_centers(self, img_w, img_h):
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        if self.results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in self.results.multi_face_landmarks[0].landmark
                ])
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            self.r_iris = np.array([l_cx, l_cy], dtype=np.int32)
            self.l_iris = np.array([r_cx, r_cy], dtype=np.int32)

    def compute_facial_distances(self, frame_debug, x, y):
        # Compute and display head position (considering nose landmark)
        p0 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        p1 = (int(self.nose_2d[0] + y * 10), int(self.nose_2d[1] - x * 10))
        cv2.line(frame_debug, p0, p1, BLACK, 1)

        # Display
        frame_debug, mouth_h = distance_between_two_lm(frame_debug, self.l_mouth, self.r_mouth, WHITE)
        frame_debug, mouth_v = distance_between_two_lm(frame_debug, self.upper_lip, self.lower_lip, WHITE)
        frame_debug, l_eyebrow_iris = distance_between_two_lm(frame_debug, self.l_eyebrow, self.l_iris, RED)
        frame_debug, l_iris_mouth = distance_between_two_lm(frame_debug, self.l_iris, self.l_mouth, GREEN)
        frame_debug, l_eyebrow_mouth = distance_between_two_lm(frame_debug, self.l_eyebrow, self.l_mouth, BLUE)
        frame_debug, r_eyebrow_iris = distance_between_two_lm(frame_debug, self.r_eyebrow, self.r_iris, RED)
        frame_debug, r_iris_mouth = distance_between_two_lm(frame_debug, self.r_mouth, self.r_iris, GREEN)
        frame_debug, r_eyebrow_mouth = distance_between_two_lm(frame_debug, self.r_eyebrow, self.r_mouth, BLUE)
        frame_debug, eye_axis = distance_between_two_lm(frame_debug, self.l_iris, self.r_iris, BLUE)

        self.facial_distances.append(mouth_h)
        self.facial_distances.append(mouth_v)
        self.facial_distances.append(l_eyebrow_iris)
        self.facial_distances.append(l_iris_mouth)
        self.facial_distances.append(l_eyebrow_mouth)
        self.facial_distances.append(r_eyebrow_iris)
        self.facial_distances.append(r_iris_mouth)
        self.facial_distances.append(r_eyebrow_mouth)
        self.facial_distances.append(eye_axis)

    def compute_facial_expression_masks(self, frame, frame_debug):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        p0 = (int(self.corrugator0[0]), int(self.corrugator0[1]))
        p2 = (int(self.corrugator1[0]), int(self.corrugator1[1]))
        p1 = (p2[0], p0[1])
        p3 = (p0[0], p2[1])
        points = np.array([p0, p1, p2, p3], np.int32)
        cv2.polylines(frame_debug, [points], True, YELLOW, 1)
        mask_corrugator = np.zeros_like(gray)
        cv2.fillPoly(mask_corrugator, [points], 255)
        self.facial_masks.append(mask_corrugator)

        p0 = (int(self.l_smile_fold0[0]), int(self.l_smile_fold0[1]))
        p1 = (int(self.l_smile_fold1[0]), int(self.l_smile_fold1[1]))
        p2 = (int(self.l_smile_fold2[0]), int(self.l_smile_fold2[1]))
        p3 = (int(self.l_smile_fold3[0]), int(self.l_smile_fold3[1]))
        points = np.array([p0, p1, p2, p3], np.int32)
        cv2.polylines(frame_debug, [points], True, YELLOW, 1)
        mask_l_smile_fold = np.zeros_like(gray)
        cv2.fillPoly(mask_l_smile_fold, [points], 255)
        self.facial_masks.append(mask_l_smile_fold)

        p0 = (int(self.r_smile_fold0[0]), int(self.r_smile_fold0[1]))
        p1 = (int(self.r_smile_fold1[0]), int(self.r_smile_fold1[1]))
        p2 = (int(self.r_smile_fold2[0]), int(self.r_smile_fold2[1]))
        p3 = (int(self.r_smile_fold3[0]), int(self.r_smile_fold3[1]))
        points = np.array([p0, p1, p2, p3], np.int32)
        cv2.polylines(frame_debug, [points], True, YELLOW, 1)
        mask_r_smile_fold = np.zeros_like(gray)
        cv2.fillPoly(mask_r_smile_fold, [points], 255)
        self.facial_masks.append(mask_r_smile_fold)

        p0 = (int(self.l_eye_bbox0[0]), int(self.l_eye_bbox0[1]))
        p1 = (int(self.l_eye_bbox1[0]), int(self.l_eye_bbox1[1]))
        p2 = (int(self.l_eye_bbox2[0]), int(self.l_eye_bbox2[1]))
        p3 = (int(self.l_eye_bbox3[0]), int(self.l_eye_bbox3[1]))
        points = np.array([p0, p1, p2, p3], np.int32)
        cv2.polylines(frame_debug, [points], True, YELLOW, 1)
        mask_l_eye = np.zeros_like(gray)
        cv2.fillPoly(mask_l_eye, [points], 255)
        self.facial_masks.append(mask_l_eye)

        p0 = (int(self.r_eye_bbox0[0]), int(self.r_eye_bbox0[1]))
        p1 = (int(self.r_eye_bbox1[0]), int(self.r_eye_bbox1[1]))
        p2 = (int(self.r_eye_bbox2[0]), int(self.r_eye_bbox2[1]))
        p3 = (int(self.r_eye_bbox3[0]), int(self.r_eye_bbox3[1]))
        points = np.array([p0, p1, p2, p3], np.int32)
        cv2.polylines(frame_debug, [points], True, YELLOW, 1)
        mask_r_eye = np.zeros_like(gray)
        cv2.fillPoly(mask_r_eye, [points], 255)
        self.facial_masks.append(mask_r_eye)

    def save_relevant_lm(self, idx, lm, x, y):
        if idx == NOSE_LANDMARK:
            self.nose_2d = (x, y)
            self.nose_3d = (x, y, lm.z)
        elif idx == RIGHT_EYEBROW_LANDMARK:
            self.r_eyebrow = (x, y)
        elif idx == LEFT_EYEBROW_LANDMARK:
            self.l_eyebrow = (x, y)
        elif idx == LEFT_MOUTH_LANDMARK or idx == L_SMILE_FOLD_2:
            self.l_mouth = (x, y)
            self.l_smile_fold3 = (x, y)
        elif idx == RIGHT_MOUTH_LANDMARK or idx == R_SMILE_FOLD_3:
            self.r_mouth = (x, y)
            self.r_smile_fold3 = (x, y)
        elif idx == UPPER_CENTRAL_LIP_LANDMARK:
            self.upper_lip = (x, y)
        elif idx == LOWER_CENTRAL_LIP_LANDMARK:
            self.lower_lip = (x, y)
        elif idx == NASAL_WRINKLES_0:
            self.corrugator0 = (x, y)
        elif idx == NASAL_WRINKLES_1:
            self.corrugator1 = (x, y)
        elif idx == R_SMILE_FOLD_0:
            self.r_smile_fold0 = (x, y)
        elif idx == R_SMILE_FOLD_1:
            self.r_smile_fold1 = (x, y)
        elif idx == R_SMILE_FOLD_2:
            self.r_smile_fold2 = (x, y)
        elif idx == L_SMILE_FOLD_1:
            self.l_smile_fold0 = (x, y)
        elif idx == L_SMILE_FOLD_0:
            self.l_smile_fold1 = (x, y)
        elif idx == L_SMILE_FOLD_3:
            self.l_smile_fold2 = (x, y)
        elif idx == L_EYE_BBOX_0:
            self.l_eye_bbox0 = (x, y)
        elif idx == L_EYE_BBOX_1:
            self.l_eye_bbox1 = (x, y)
        elif idx == L_EYE_BBOX_2:
            self.l_eye_bbox2 = (x, y)
        elif idx == L_EYE_BBOX_3:
            self.l_eye_bbox3 = (x, y)
        elif idx == R_EYE_BBOX_0:
            self.r_eye_bbox0 = (x, y)
        elif idx == R_EYE_BBOX_1:
            self.r_eye_bbox1 = (x, y)
        elif idx == R_EYE_BBOX_2:
            self.r_eye_bbox2 = (x, y)
        elif idx == R_EYE_BBOX_3:
            self.r_eye_bbox3 = (x, y)

    def extract_frame_features(self, frame):
        frame_debug = frame.copy()
        img_h, img_w = frame_debug.shape[:2]
        self.results = self.faceMesh.process(frame_debug)
        self.get_iris_centers(img_w, img_h)
        self.facial_masks = []
        self.facial_distances = []
        self.head_position = [0] * 6

        face_3d = []
        face_2d = []

        if self.results.multi_face_landmarks is None:
            return frame, self.head_position, self.facial_distances, self.facial_masks

        for landmarks in self.results.multi_face_landmarks:
            for idx, lm in enumerate(landmarks.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
                self.save_relevant_lm(idx, lm, x, y)

            x, y, z = get_rotation_angles(face_2d, face_3d, img_w, img_h)

            direction_text = looking_direction(x, y)
            frame_debug = frame_text_direction_and_angles(frame_debug, direction_text, img_w, x, y, z)

            self.head_position[0] += self.nose_3d[0]
            self.head_position[1] += self.nose_3d[1]
            self.head_position[2] += self.nose_3d[2]
            self.head_position[3] += x
            self.head_position[4] += y
            self.head_position[5] += z
            self.compute_facial_distances(frame_debug, x, y)
            self.compute_facial_expression_masks(frame, frame_debug)

        return frame_debug, self.head_position, self.facial_distances, self.facial_masks
