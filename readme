🎯 Focus Monitor
A real-time focus detection system that alerts you when you're distracted by your phone.
How It Works
The system uses your webcam to monitor two things simultaneously:

Phone Detection — YOLOv8 detects if a phone is visible in the frame
Head Pose Estimation — MediaPipe tracks your face landmarks and calculates the direction your head is pointing

If both conditions are true at the same time (phone detected + head tilted toward it), an alarm sound plays immediately to snap you back to focus.
Tech Stack

YOLOv8 — Real-time object detection
MediaPipe Face Mesh — 468-point facial landmark tracking
OpenCV — Webcam capture and video processing
Pygame — Audio alert system
Python 3.11

Features

Real-time head pose estimation (yaw & pitch angles)
Buffered detection to avoid false alarms
Visual overlay showing phone and gaze status
Cooldown system between alerts

Setup
bashpip install mediapipe ultralytics pygame opencv-python
bashpython focus_app.py
Configuration
pythonYAW_THRESHOLD = 25    # Head turn sensitivity (degrees)
PITCH_THRESHOLD = 20  # Head tilt sensitivity (degrees)
ALERT_COOLDOWN = 3.0  # Seconds between alerts
