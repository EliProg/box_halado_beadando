import cv2
import mediapipe as mp
import math 

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

punch_threshold = 0.35 
reset_threshold = 0.15
raised_threshold = 0.2


r_arm_ready = True 
l_arm_ready = True
r_jab_count = 0 
l_jab_count = 0

def get_landmark_coordinates(landmark, landmark_i):
    lm = landmark.landmark[landmark_i]
    return lm.x, lm.y


def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while cap.isOpened():
    success, image = cap.read()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
 
    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        
        r_shoulder_x, r_shoulder_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        r_wrist_x, r_wrist_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
        
        l_shoulder_x, l_shoulder_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        l_wrist_x, l_wrist_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
    
        r_dist = calculate_distance(r_shoulder_x, r_shoulder_y, r_wrist_x, r_wrist_y)
        l_dist = calculate_distance(l_shoulder_x, l_shoulder_y, l_wrist_x, l_wrist_y)

        
        if r_arm_ready:
    
            if r_dist > punch_threshold and r_wrist_y < (r_shoulder_y + raised_threshold):
                r_jab_count += 1 
                r_arm_ready = False 
        else: 
            
            if r_dist < reset_threshold:
                r_arm_ready = True
        
        
        if l_arm_ready:
            if l_dist > punch_threshold and l_wrist_y < (l_shoulder_y + raised_threshold):
                l_jab_count += 1 
                l_arm_ready = False
        else:
            if l_dist < reset_threshold:
                l_arm_ready = True
        
 
    cv2.putText(image, f'R Jabs: {r_jab_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(image, f'L Jabs: {l_jab_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    

    cv2.imshow('Pose Estimation', image) 
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
