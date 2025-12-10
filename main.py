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

cap = cv2.VideoCapture(2)


punch_threshold = 0.35  
reset_threshold = 0.15  
raised_threshold = 0.2  

#felütések
r_upper_ready = True
l_upper_ready = True
r_upper_count = 0
l_upper_count = 0

# egyenes ütések
r_arm_ready = True 
l_arm_ready = True
r_jab_count = 0 
l_jab_count = 0

def get_landmark_coordinates(landmark, landmark_i):
    lm = landmark.landmark[landmark_i]
    return lm.x, lm.y

def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

if not cap.isOpened():
    print("Camera not found.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
 
    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        
        nose_x, nose_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.NOSE.value)

        r_shoulder_x, r_shoulder_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        r_elbow_x, r_elbow_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value) # Need Elbow
        r_wrist_x, r_wrist_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
        
        l_shoulder_x, l_shoulder_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        l_elbow_x, l_elbow_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value) # Need Elbow
        l_wrist_x, l_wrist_y = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
    
        # Calculate Distances for Jabs
        r_dist = calculate_distance(r_shoulder_x, r_shoulder_y, r_wrist_x, r_wrist_y)
        l_dist = calculate_distance(l_shoulder_x, l_shoulder_y, l_wrist_x, l_wrist_y)

        # ==========================================================
        # 🥊 JAB DETECTION (Horizontal Extension)
        # ==========================================================
        
        # RIGHT HAND DATA -> LEFT COUNTER (Mirror)
        if r_arm_ready:
            if r_dist > punch_threshold and r_wrist_y < (r_shoulder_y + raised_threshold):
                l_jab_count += 1
                r_arm_ready = False 
        else: 
            if r_dist < reset_threshold:
                r_arm_ready = True
        
        # LEFT HAND DATA -> RIGHT COUNTER (Mirror)
        if l_arm_ready:
            if l_dist > punch_threshold and l_wrist_y < (l_shoulder_y + raised_threshold):
                r_jab_count += 1
                l_arm_ready = False
        else:
            if l_dist < reset_threshold:
                l_arm_ready = True

        if r_upper_ready:
            if (r_wrist_y < nose_y and 
                r_elbow_y > r_shoulder_y and 
                r_dist < punch_threshold): # Ensure it's not a jab
                
                l_upper_count += 1
                r_upper_ready = False
        else:
           
            if r_wrist_y > (nose_y + 0.15):
                r_upper_ready = True

       
        if l_upper_ready:
            if (l_wrist_y < nose_y and 
                l_elbow_y > l_shoulder_y and 
                l_dist < punch_threshold):
                
                r_upper_count += 1
                l_upper_ready = False
        else:
            if l_wrist_y > (nose_y + 0.15):
                l_upper_ready = True

  
    cv2.putText(image, f'R Jabs: {r_jab_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f'L Jabs: {l_jab_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    

    cv2.putText(image, f'R Uppers: {r_upper_count}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f'L Uppers: {l_upper_count}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    cv2.imshow('Boxing Trainer', image) 
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()