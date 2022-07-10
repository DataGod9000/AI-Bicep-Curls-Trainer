# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:27:43 2022

@author: biyin
"""
import csv
import numpy as np
import cv2 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

###############################################################################################################
#GETTING LANDMARK

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
###############################################################################################################


    
#CALCULATING ANGLES

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

#Assiging respective X and Y coordinates to shoulder, elbow and wrist

l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

#capturing of landmark 

pose_coord_loc = len(results.pose_landmarks.landmark)

pose_landmarks = ['body class']

for val in range(1, pose_coord_loc+1):
    pose_landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val),]


exercise_name = 'Left Bicep Curl'

# Video Feed
cap = cv2.VideoCapture(0)

# Curl counter variables
l_counter = 0 
r_counter = 0
l_stage = None
r_stage = None

#Flex
l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
l_per = np.interp(l_angle, (30,170), (0, 100))
l_bar = np.interp(l_angle, (30,170), (120, 400))
colour = (75,75,216)

r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
r_per = np.interp(r_angle, (30,170), (0, 100))
r_bar = np.interp(r_angle, (30,170), (120, 400))
colour = (75,75,216)


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Detecting and Rendering Landmarks by MediaPipe
        # Recolor image to RGB
        # MediaPipe processes RGB img only
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection (MediaPipe)
        results = pose.process(image)
    
        # Recolor back to BGR (Or your Video will be Blue and Grey)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks 
        # Prevent destroying entire feed if landmarks are not extracted as expected 
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            
            # Calculate angle
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            # Visualize angle and Logics
            cv2.putText(image, str(l_angle), 
                           tuple(np.multiply(l_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(r_angle), 
                           tuple(np.multiply(r_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # left Reps Counter Logic           
            
            if l_angle > 160:
                l_stage = "down"
                colour = (75,75,216)
                
            if l_angle >30:
                colour = (75,75,216)
                
            if l_angle < 30 and l_stage =='down':
                l_stage="up"
                l_counter +=1
                colour = (154,250,0)
                print(l_counter)
            
            # right Reps Counter Logic          
            
            if r_angle > 160:
                r_stage = "down"
                colour = (75,75,216)
                
            if r_angle >30:
                colour = (75,75,216)
                
            if r_angle < 30 and r_stage =='down':
                r_stage="up"
                r_counter +=1
                colour = (154,250,0)
                print(r_counter)
                
            # left Flexion Logic   
            l_per = np.interp(l_angle, (30,170), (100, 0))
            print(l_angle, l_per)
            
            l_bar = np.interp(l_angle, (30,170), (120, 400))

            # left Flexion Logic   
            
            r_per = np.interp(r_angle, (30,170), (100, 0))
            print(r_angle, r_per)
            
            r_bar = np.interp(r_angle, (30,170), (120, 400))
            
            #collect coords and export to csv
            
            #get all the coordinates from the landmarks
            pose_workout = results.pose_landmarks.landmark
            
            #extracting all the coords and converting it to an array
            pose_workout_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_workout]).flatten()
            
            #covert the array to a list
            pose_workout_row = list(pose_workout_row)
            
            #concatenate rows together
            #when there are more than just one
            total_row = pose_workout_row
            total_row.insert(0, exercise_name)
            
            #create a new CSV writer for pose landmarks
            #mode='a' as append
            with open('C:/Users/get gd nub/Desktop/AI Personal trainer/coord_export_leftbicep.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(total_row)
            
            
        except:
            pass
        
        # VISUALISATION ON SCREEN (BOXES, TEXT, AND BARS)
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (147,20,255), -1)
        cv2.rectangle(image, (400, 0), (680,73), (147,20,255), -1)
        
        #left flexion bar
        cv2.rectangle(image, (580,400), (630,120), colour, -1)
        cv2.rectangle(image, (580,int(l_bar)), (630,120), (255,255,255), -1)
        
        #right flexion bar
        cv2.rectangle(image, (10,400), (60,120), colour, -1)
        cv2.rectangle(image, (10,int(r_bar)), (60,120), (255,255,255), -1)
        
        # right Rep data
        cv2.putText(image, 'REPS', (15,18), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(r_counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # left Rep data
        cv2.putText(image, 'REPS', (450,18), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(l_counter), 
                    (445,60), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        
        #left flex percentage
        cv2.putText(image, f'{int(l_per)}%', (580,460), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colour, 2)
        
        #right flex percentage
        cv2.putText(image, f'{int(r_per)}%', (10,460), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colour, 2)
        
        #right Stage data
        cv2.putText(image, 'STAGE', (65,18),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, r_stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)

        #left Stage data
        cv2.putText(image, 'STAGE', (500,18),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, l_stage, 
                    (495,60), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        print (l_angle, l_per, r_angle, r_per)
        
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()