import cv2, time
from eyetracker import EyeTracker
import numpy as np

cap = cv2.VideoCapture(1)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
DIM = (2880, 1800)


out = cv2.VideoWriter('output/output.mov', cv2.VideoWriter_fourcc(*'mp4v'), fps // 2, (DIM[0], 2 * DIM[1]), True)

top_left_angles = []
bottom_right_angles = []

article = cv2.imread('assets/article.png')
article = cv2.resize(article, DIM)

t = 0
while cap.isOpened():
    ret, frame = cap.read()
    img = article.copy()
    if not ret:
        break
    frame = cv2.resize(frame, DIM)
    frame = cv2.flip(frame,1)

    tracker = EyeTracker(DIM)
    frame = tracker.detect(frame, draw_mesh=True)

    out_image = np.zeros([2 * DIM[1], DIM[0], 3], dtype=np.uint8)

    if t < 30:
        img = cv2.putText(img, f'Stare at top-left corner in: {(30 - t) // 10 + 1}..', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)
    elif t < 40:
        img = np.zeros([DIM[1], DIM[0], 3], dtype=np.uint8)
        img.fill(255)
        top_left_angles.append(tracker.gaze_angle)
        top_left = np.mean(np.array(top_left_angles), axis=0)
    elif t < 70:
        img = cv2.putText(img, f'Stare at bottom-right corner in: {(70 - t) // 10 + 1}..', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)
    elif t < 80:
        img = np.zeros([DIM[1], DIM[0], 3], dtype=np.uint8)
        img.fill(255)
        bottom_right_angles.append(tracker.gaze_angle)
        bottom_right = np.mean(np.array(bottom_right_angles), axis=0)
        prev = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
    else:
        prev.pop(0)
        prev.append(tracker.gaze_angle)
        angle = np.mean(np.array(prev), axis=0)
        prev[-1] = angle
        max_x, max_y = bottom_right - top_left
        angle_x, angle_y =  angle - top_left

        x = int(angle_x / max_x * DIM[0])
        y = int(angle_y / max_y * DIM[1])

        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (x, y), 350, (255, 255, 255), -1)

        blur_img = cv2.blur(img, [30, 30])
        img = np.where(mask > 0, blur_img, img)


    cv2.imshow('Image', img)
    out_image[:DIM[1], :,:] = img
    out_image[DIM[1]:, :,:] = frame
    
    out.write(out_image)

    t += 1
    if cv2.waitKey(1000 // fps) == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()