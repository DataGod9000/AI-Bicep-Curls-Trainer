
# AI Bicep Curls Trainer

This project uses Mediapipe, OpenCV and Machine Learning to:

1) Predict if the user is doing left or right bicep curl
2) Counts reps
2) Detect whether or not the user is achieving full flexion when doing a bicep curl
3) Detect whether or not user is performing the proper form. (Bad reps - reps that didn't achieve full flexion or with elbow moving over a certain extend - are not counted)


## Demo
## Work Flow

1) Import and embedd Mediapipe into OpenCV to map and track poses
2) Obtain landmark coordinates of poses and export them as CSV files
3) Calculate Angle and write underlying logic to detect good and bad forms
4) Train models with coordinates exported
5) Integrate model into OpenCV and Mediapipe to predict left or right bicep curl
## Model

Model of Choice: Random Forest

Model Accuracy:

## Limitations and Futureworks

Limitations:

1) The current AI trainer only limits to bicep curls
2) Flexion Bar do not work as well when the entire arm is covered 

Future Works:

1) Explore detection of workouts with CNN
2) Include more execises like Deadlifts, Squats, and Benchpresses
3) Include an indicator to inform users how many degrees they are off from achieving good form