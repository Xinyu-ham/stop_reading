import mediapipe as mp
import numpy as np
import cv2
if __name__ == '__main__':
    from utils import LEFT_EYE, RIGHT_EYE, LEFT_PUPIL, RIGHT_PUPIL, get_center, get_gaze_angle
else:
    from .utils import LEFT_EYE, RIGHT_EYE, LEFT_PUPIL, RIGHT_PUPIL, get_center, get_gaze_angle

face_mesh = mp.solutions.face_mesh

class EyeTracker:
    def __init__(self, dim: tuple[int]):
        self.dim = dim
        self.face_mesh = face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
            refine_landmarks=True,
            max_num_faces=1,
            static_image_mode=False)
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.landmarks = None

    def detect(self, frame: np.ndarray, draw_mesh=True) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.face_mesh.process(frame).multi_face_landmarks
        
        if landmarks:
            self.landmarks = landmarks[0]
            frame.flags.writeable = True
            if draw_mesh:
                self.draw_mesh(frame)

            self.save_eye_landmarks()
            frame = self.draw_eyes(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def draw_mesh(self, frame: np.ndarray) -> np.ndarray:
        self.drawing.draw_landmarks(
            frame, 
            self.landmarks, 
            face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=self.drawing.DrawingSpec(thickness=2, circle_radius=1),
            connection_drawing_spec=self.styles.get_default_face_mesh_iris_connections_style()
        )

    def get_landmarks(self, indices: list[int]) -> list[tuple]:
        landmarks = self.landmarks.landmark
        return [np.array([int(landmark.x * self.dim[0]), int(landmark.y * self.dim[1])]) for i, landmark in enumerate(landmarks) if i in indices]
    
    def save_eye_landmarks(self):
        self.left_eye = get_center(self.get_landmarks(LEFT_EYE))
        self.right_eye = get_center(self.get_landmarks(RIGHT_EYE))
        self.left_pupil = get_center(self.get_landmarks(LEFT_PUPIL))
        self.right_pupil = get_center(self.get_landmarks(RIGHT_PUPIL))
        self.gaze_angle = get_gaze_angle(self.left_eye, self.right_eye, self.left_pupil, self.right_pupil)

    def draw_eyes(self, frame):
        for eye in (self.left_eye, self.right_eye):
            frame = cv2.circle(frame, eye, 5, (0, 255, 0), -1)

        for pupil in (self.left_pupil, self.right_pupil):
            frame = cv2.circle(frame, pupil, 5, (0, 0, 255), -1)
        frame = cv2.arrowedLine(frame, self.left_eye, (int(self.left_eye[0] + 2 * self.gaze_angle[0]), int(self.left_eye[1] +  2 * self.gaze_angle[1])), (255, 0, 0), 2)
        frame = cv2.arrowedLine(frame, self.right_eye, (int(self.right_eye[0] + 2 * self.gaze_angle[0]), int(self.right_eye[1] + 2 * self.gaze_angle[1])), (255, 0, 0), 2)
        return frame
if __name__ == '__main__':
    import cv2
    dim = (2880, 1800)
    img = cv2.imread('../assets/topleft.png')
    img = cv2.resize(img, dim)
    

    tracker = EyeTracker(dim)
    img = tracker.detect(img, draw=False)
    img = tracker.draw_eyes(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
