import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )

cap = cv2.VideoCapture(0)

def get_landmark_coordinates(landmark, landmark_i):
    lm = landmark.landmark[landmark_i]
    return lm.x, lm.y

distance_threshold = 0.2

jab_count = 0

while cap.isOpened():

    jab_detected_prev = False
    jab_detected_now = False
    arm_extended = False
    arm_retracted = True
    success, image = cap.read()

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False

    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

   
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        shoulder_x, _ = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        wrist_x, _ = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)

        if shoulder_x and wrist_x and arm_extended == False:
            if wrist_x < (shoulder_x - distance_threshold):
                jab_detected_now = True
                arm_extended = True
                arm_retracted = False
        else :
            jab_detected_now = False

        if wrist_x >= (shoulder_x - distance_threshold):
            arm_retracted = True
            arm_extended = False

    if jab_detected_now and not jab_detected_prev:
        jab_count += 1

    jab_detected_prev = jab_detected_now


    cv2.putText(image, 
            f'Jabs: {jab_count}', 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 255, 255), 
            3)
    
    cv2.imshow('MediaPipe Pose Estimation', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
pose.close()