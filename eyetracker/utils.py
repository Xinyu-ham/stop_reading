import numpy as np

LEFT_EYE = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_PUPIL = [468, 469, 470, 471, 472]
RIGHT_PUPIL = [473, 474, 475, 476, 477]



def get_center(landmarks: list[tuple]) -> tuple[np.ndarray]:
    x, y = np.mean(landmarks, axis=0)
    return int(x), int(y)

def get_gaze_angle_per_eye(eye: tuple[int], pupil: tuple[int]) -> float:
    eye = np.array(eye)
    pupil = np.array(pupil)
    return pupil - eye

def get_gaze_angle(left_eye, right_eye, left_pupil, right_pupil):
    left_x, left_y = get_gaze_angle_per_eye(left_eye, left_pupil)
    right_x, right_y = get_gaze_angle_per_eye(right_eye, right_pupil)
    return left_x + right_x / 2, left_y + right_y / 2