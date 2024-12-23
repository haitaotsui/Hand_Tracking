# -------------------------------------
# Create: Haitao Xu
# Date  : 10/26/2024
# -------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# -------------------------------
# Initialize MediaPipe Hand Model
# -------------------------------

# Access the hands solution from MediaPipe
mp_hands = mp.solutions.hands

# Access the drawing utilities from MediaPipe for visualization
mp_drawing = mp.solutions.drawing_utils

# Initialize the MediaPipe Hands model with specified parameters:
# - min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful
# - min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------------------------------
# Initialize OpenCV Video Capture
# -------------------------------

# Open a connection to the default camera (usually the first webcam)
cap = cv2.VideoCapture(0)

# -------------------------------
# Define Gesture Detection Function
# -------------------------------

def is_victory_gesture(hand_landmarks, frame_width, frame_height):
    """
    Determines if the detected hand landmarks correspond to a "V" (Victory) gesture.

    Args:
        hand_landmarks: The landmarks of the detected hand provided by MediaPipe.
        frame_width: The width of the video frame in pixels.
        frame_height: The height of the video frame in pixels.

    Returns:
        bool: True if the "V" gesture is detected, False otherwise.
    """
    
    # Extract the (x, y) coordinates of each landmark and convert them to pixel values
    keypoints = [
        (int(p.x * frame_width), int(p.y * frame_height)) 
        for p in hand_landmarks.landmark
    ]
    
    # -------------------------------
    # Identify Specific Landmarks
    # -------------------------------
    
    # Fingertip landmarks (indices based on MediaPipe Hands documentation)
    index_tip = keypoints[8]     # Tip of the index finger
    middle_tip = keypoints[12]   # Tip of the middle finger
    
    # MCP (Metacarpophalangeal) joint landmarks for index and middle fingers
    index_mcp = keypoints[5]     # MCP joint of the index finger
    middle_mcp = keypoints[9]    # MCP joint of the middle finger

    # -------------------------------
    # Determine if Index and Middle Fingers are Straight
    # -------------------------------
    
    # A finger is considered straight if the y-coordinate of the tip is less than the y-coordinate of the MCP joint
    # (In image coordinates, y increases from top to bottom)
    is_index_straight = index_tip[1] < index_mcp[1]
    is_middle_straight = middle_tip[1] < middle_mcp[1]

    # -------------------------------
    # Identify Other Finger Tips
    # -------------------------------
    
    # Landmarks for thumb, ring, and pinky fingertips
    thumb_tip = keypoints[4]      # Tip of the thumb
    ring_tip = keypoints[16]      # Tip of the ring finger
    pinky_tip = keypoints[20]     # Tip of the pinky finger

    # -------------------------------
    # Determine if Thumb, Ring, and Pinky Fingers are Closed
    # -------------------------------
    
    # A finger is considered closed if the y-coordinate of the tip is greater than the y-coordinate of the MCP joint
    is_thumb_closed = thumb_tip[1] > index_mcp[1]
    is_ring_closed = ring_tip[1] > middle_mcp[1]
    is_pinky_closed = pinky_tip[1] > middle_mcp[1]

    # -------------------------------
    # Final Condition for "V" Gesture
    # -------------------------------
    
    # The "V" gesture is identified when:
    # - Index and middle fingers are straight
    # - Thumb, ring, and pinky fingers are closed
    return (
        is_index_straight and 
        is_middle_straight and 
        is_thumb_closed and 
        is_ring_closed and 
        is_pinky_closed
    )

# -------------------------------
# Define Fireworks Drawing Function
# -------------------------------

def draw_fireworks(frame):
    """
    Draws a fireworks-like effect on the given frame by randomly placing colored circles.

    Args:
        frame: The current video frame where the fireworks effect will be drawn.

    Returns:
        frame: The modified frame with fireworks drawn on it.
    """
    
    # Get the dimensions of the frame
    h, w, _ = frame.shape
    
    # Generate 50 random points to simulate fireworks
    for _ in range(50):
        # Random x-coordinate within the frame width
        x = random.randint(0, w)
        
        # Random y-coordinate within the frame height
        y = random.randint(0, h)
        
        # Random color in BGR format
        color = (
            random.randint(0, 255),  # Blue component
            random.randint(0, 255),  # Green component
            random.randint(0, 255)   # Red component
        )
        
        # Draw a filled circle (representing a firework particle) at the random position
        # -1 thickness indicates a filled circle
        cv2.circle(frame, (x, y), random.randint(5, 10), color, -1)
    
    # Return the frame with the fireworks drawn
    return frame

# -------------------------------
# Main Execution Loop
# -------------------------------

while True:
    # Capture a single frame from the webcam
    ret, frame = cap.read()
    
    # If frame capture was unsuccessful, exit the loop
    if not ret:
        break

    # -------------------------------
    # Preprocess the Frame
    # -------------------------------
    
    # Flip the frame horizontally to create a mirror image (optional for better user experience)
    frame = cv2.flip(frame, 1)

    # Convert the BGR image (default in OpenCV) to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # Process the Frame with MediaPipe Hands
    # -------------------------------
    
    # Perform hand detection and landmark recognition
    results = hands.process(rgb_frame)

    # -------------------------------
    # Handle Detected Hands
    # -------------------------------
    
    # Check if any hands are detected in the frame
    if results.multi_hand_landmarks:
        # Iterate over each detected hand (supports multiple hands)
        for hand_landmarks in results.multi_hand_landmarks:
            # -------------------------------
            # Visualize Hand Landmarks
            # -------------------------------
            
            # Draw the hand landmarks and connections on the original frame
            # - frame: The image on which to draw
            # - hand_landmarks: The detected landmarks of the current hand
            # - mp_hands.HAND_CONNECTIONS: The predefined connections between landmarks for drawing
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            # -------------------------------
            # Detect "V" (Victory) Gesture
            # -------------------------------
            
            # Call the gesture detection function with current hand landmarks and frame dimensions
            if is_victory_gesture(hand_landmarks, frame.shape[1], frame.shape[0]):
                # If "V" gesture is detected, overlay text on the frame
                cv2.putText(
                    frame, 
                    "Victory Gesture Detected!",          # Text to display
                    (50, 50),                            # Position (x, y) in pixels
                    cv2.FONT_HERSHEY_SIMPLEX,            # Font type
                    1,                                   # Font scale (size)
                    (0, 255, 0),                         # Text color in BGR (Green)
                    2                                    # Thickness of the text
                )
                
                # -------------------------------
                # Trigger Fireworks Effect
                # -------------------------------
                
                # Call the fireworks drawing function to add fireworks to the frame
                frame = draw_fireworks(frame)

    # -------------------------------
    # Display the Processed Frame
    # -------------------------------
    
    # Show the frame in a window titled "Hand Gesture Recognition with Fireworks"
    cv2.imshow("Hand Gesture Recognition with Fireworks", frame)

    # -------------------------------
    # Exit Mechanism
    # -------------------------------
    
    # Wait for 1 millisecond for a key press
    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Release Resources and Close Windows
# -------------------------------

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
