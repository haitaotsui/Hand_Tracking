import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # MediaPipe for hand tracking

# Initialize MediaPipe hand tracking modules
mp_hands = mp.solutions.hands  # Hands module for detecting and tracking hands
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for rendering landmarks and connections

# Configure the hand tracking solution
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for dynamic video input
    max_num_hands=2,          # Maximum number of hands to detect at the same time
    min_detection_confidence=0.5,  # Minimum confidence value for the initial hand detection
    min_tracking_confidence=0.5    # Minimum confidence value for tracking hands after detection
)

# Open the webcam (camera index 0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Inform the user how to exit the program
print("Press 'q' to exit the program.")

# Main video capture loop
while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")  # Handle the case where the frame capture fails
        break

    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to detect and track hands
    results = hands.process(rgb_frame)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # Loop through all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks (red circles) and connections (thick green lines) on the frame
            mp_drawing.draw_landmarks(
                frame,                       # The image/frame on which landmarks will be drawn
                hand_landmarks,              # The hand landmarks detected by MediaPipe
                mp_hands.HAND_CONNECTIONS,   # Predefined connections between hand landmarks
                mp_drawing.DrawingSpec(      # Style for landmarks (red dots)
                    color=(0, 0, 255),       # RGB color (red)
                    thickness=5,             # Thickness of the dots
                    circle_radius=5          # Radius of the dots
                ),
                mp_drawing.DrawingSpec(      # Style for connections (green lines)
                    color=(0, 255, 0),       # RGB color (green)
                    thickness=4,             # Thickness of the lines
                    circle_radius=2          # Unused for lines but required
                )
            )

    # Display the frame with the drawn landmarks and connections
    cv2.imshow('Hand Tracking', frame)

    # Wait for 1 ms and check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
