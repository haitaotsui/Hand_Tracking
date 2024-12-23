# -------------------------------------
# Create: Haitao Xu
# Date  : 10/26/2024
# -------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

# -------------------------------------
# Initialize MediaPipe Hands
# -------------------------------------

# Access the MediaPipe Hands solution, which provides real-time hand tracking.
mp_hands = mp.solutions.hands

# Access the drawing utilities from MediaPipe for visualizing hand landmarks.
mp_draw = mp.solutions.drawing_utils

# Initialize the MediaPipe Hands model with specific parameters:
# - static_image_mode=False: The solution treats the input images as a video stream.
# - max_num_hands=1: Limits the detection to a single hand.
# - min_detection_confidence=0.7: Sets the minimum confidence threshold for hand detection.
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -------------------------------------
# Define Finger Tip Indices
# -------------------------------------

# List of landmark indices corresponding to the fingertips.
FINGER_TIPS = [4, 8, 12, 16, 20]
# Explanation:
# - 4: Thumb tip
# - 8: Index finger tip
# - 12: Middle finger tip
# - 16: Ring finger tip
# - 20: Pinky finger tip

# -------------------------------------
# Define Gesture Enumeration
# -------------------------------------

# A class to represent different types of gestures.
class Gesture:
    V = "V Gesture"            # Represents the "Victory" or "Peace" gesture.
    OK = "OK Gesture"          # Represents the "OK" gesture.
    THUMBS_UP = "Thumbs Up"    # Represents the "Thumbs Up" gesture.
    UNKNOWN = "Unknown"        # Represents an unrecognized gesture.

# -------------------------------------
# Gesture Detection Function
# -------------------------------------

def detect_gesture(hand_landmarks):
    """
    Identifies the gesture being performed based on hand landmarks.

    Args:
        hand_landmarks: The landmarks of the detected hand provided by MediaPipe.

    Returns:
        A string representing the detected gesture.
    """
    # List to store the state (extended or folded) of each finger.
    fingers = []

    # ---------------------------------
    # Detect Thumb Extension
    # ---------------------------------

    # Compare the x-coordinate of the thumb tip with the thumb interphalangeal (IP) joint.
    # If the thumb tip is to the left of the IP joint (for a right hand), it's considered extended.
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)  # Thumb is extended.
    else:
        fingers.append(0)  # Thumb is folded.

    # ---------------------------------
    # Detect Extension of Other Fingers
    # ---------------------------------

    # Iterate over the indices corresponding to the tips of the index, middle, ring, and pinky fingers.
    for tip in FINGER_TIPS[1:]:
        # Compare the y-coordinate of the fingertip with the y-coordinate of the pip joint (two landmarks below the tip).
        # In image coordinates, y increases from top to bottom.
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Finger is extended.
        else:
            fingers.append(0)  # Finger is folded.

    # ---------------------------------
    # Determine the Gesture Based on Finger States
    # ---------------------------------

    # ---------------------------------
    # Thumbs Up Gesture: Only the thumb is extended.
    # ---------------------------------
    if fingers == [1, 0, 0, 0, 0]:
        return Gesture.THUMBS_UP

    # ---------------------------------
    # V Gesture: Index and middle fingers are extended.
    # ---------------------------------
    elif fingers == [0, 1, 1, 0, 0]:
        return Gesture.V

    # ---------------------------------
    # OK Gesture: Thumb and index finger form a circle, others are extended.
    # ---------------------------------
    else:
        # Calculate the Euclidean distance between the thumb tip and index finger tip.
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

        # If the distance is below a threshold and all fingers except the thumb are extended, it's an OK gesture.
        if distance < 0.05 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            return Gesture.OK

    # If none of the above conditions are met, return UNKNOWN.
    return Gesture.UNKNOWN

# -------------------------------------
# Firework Effect Class
# -------------------------------------

class Firework:
    """
    Represents a firework effect consisting of multiple particles.
    """
    def __init__(self, position):
        """
        Initializes the Firework instance.

        Args:
            position: A tuple (x, y) representing the origin of the firework.
        """
        self.position = position          # Origin position of the firework.
        self.particles = []               # List to store particles.
        self.create_particles()           # Initialize particles.

    def create_particles(self):
        """
        Generates particles for the firework effect.
        """
        for _ in range(20):  # Each firework has 20 particles.
            angle = random.uniform(0, 2 * math.pi)          # Random angle for direction.
            speed = random.uniform(2, 5)                    # Random speed.
            # Each particle is represented as a dictionary with its properties.
            self.particles.append({
                'position': list(self.position),            # Current position.
                'velocity': [math.cos(angle) * speed, math.sin(angle) * speed],  # Velocity vector.
                'size': 4,                                   # Size of the particle.
                'color': (
                    random.randint(0, 255),                  # Blue component.
                    random.randint(0, 255),                  # Green component.
                    random.randint(0, 255)                   # Red component.
                ),
                'life': 50                                   # Lifespan of the particle.
            })

    def update(self, frame):
        """
        Updates the state of each particle and renders them on the frame.

        Args:
            frame: The current video frame to draw particles on.
        """
        # Iterate over a copy of the particles list to allow safe removal.
        for particle in self.particles[:]:
            # Update particle position based on its velocity.
            particle['position'][0] += particle['velocity'][0]
            particle['position'][1] += particle['velocity'][1]

            # Apply gravity effect by incrementing the y-velocity.
            particle['velocity'][1] += 0.1

            # Decrease the lifespan of the particle.
            particle['life'] -= 1

            # Draw the particle as a filled circle on the frame.
            cv2.circle(
                frame,
                (int(particle['position'][0]), int(particle['position'][1])),
                particle['size'],
                particle['color'],
                -1  # Negative thickness indicates a filled circle.
            )

            # Remove the particle if its lifespan has ended.
            if particle['life'] <= 0:
                self.particles.remove(particle)

# -------------------------------------
# Neon Effect Class
# -------------------------------------

class NeonEffect:
    """
    Represents a neon-like pulsating circular effect.
    """
    def __init__(self, position):
        """
        Initializes the NeonEffect instance.

        Args:
            position: A tuple (x, y) representing the center of the neon effect.
        """
        self.position = position          # Center position of the neon effect.
        self.max_radius = 150             # Maximum radius during pulsation.
        self.min_radius = 100             # Minimum radius during pulsation.
        self.radius = self.min_radius     # Current radius.
        self.growing = True               # Indicates whether the radius is increasing.
        self.color = (0, 255, 255)        # Cyan color for the neon effect.
        self.thickness = 4                # Thickness of the neon ring.
        self.duration = 3.0               # Total duration of the effect in seconds.
        self.start_time = time.time()     # Timestamp when the effect was initiated.

    def update(self, frame):
        """
        Updates the neon effect's state and renders it on the frame.

        Args:
            frame: The current video frame to draw the neon effect on.

        Returns:
            bool: True if the effect is still active, False otherwise.
        """
        # Calculate the elapsed time since the effect started.
        elapsed = time.time() - self.start_time

        # If the effect duration has been exceeded, indicate that the effect should end.
        if elapsed > self.duration:
            return False  # Effect ends.

        # ---------------------------------
        # Update Radius for Pulsation
        # ---------------------------------

        if self.growing:
            self.radius += 2  # Increase radius.
            if self.radius >= self.max_radius:
                self.growing = False  # Start decreasing radius.
        else:
            self.radius -= 2  # Decrease radius.
            if self.radius <= self.min_radius:
                self.growing = True   # Start increasing radius.

        # ---------------------------------
        # Draw Neon Circular Ring
        # ---------------------------------

        # Draw the main neon circle.
        cv2.circle(
            frame,
            self.position,
            self.radius,
            self.color,
            self.thickness
        )

        # ---------------------------------
        # Add Glowing Overlay Effect
        # ---------------------------------

        # Draw additional circles to simulate a glow effect.
        for i in range(1, 4):
            overlay_color = (
                self.color[0],
                self.color[1],
                self.color[2],
                max(255 - i * 60, 0)  # Adjust alpha or brightness if desired.
            )
            cv2.circle(
                frame,
                self.position,
                self.radius + i * 10,
                self.color,
                2  # Thin circles for the glow.
            )

        return True  # Effect remains active.

# -------------------------------------
# Clap Effect Class
# -------------------------------------

class ClapEffect:
    """
    Represents a clap-like textual effect that appears briefly.
    """
    def __init__(self, position):
        """
        Initializes the ClapEffect instance.

        Args:
            position: A tuple (x, y) representing where the "WoW" text will appear.
        """
        self.position = position          # Position to display the text.
        self.start_time = time.time()     # Timestamp when the effect was initiated.
        self.duration = 1.0               # Total duration of the effect in seconds.

    def update(self, frame):
        """
        Updates the clap effect's state and renders it on the frame.

        Args:
            frame: The current video frame to draw the clap effect on.

        Returns:
            bool: True if the effect is still active, False otherwise.
        """
        # Calculate the elapsed time since the effect started.
        elapsed = time.time() - self.start_time

        # If the effect duration has been exceeded, indicate that the effect should end.
        if elapsed > self.duration:
            return False  # Effect ends.

        # Calculate the transparency factor (alpha) based on elapsed time.
        alpha = 1.0 - (elapsed / self.duration)

        # Calculate the size scaling factor for a diminishing effect.
        size = int(100 * alpha)

        # If the size is still positive, render the text.
        if size > 0:
            cv2.putText(
                frame,
                "WoW",                              # Text to display.
                self.position,                      # Position (x, y) in pixels.
                cv2.FONT_HERSHEY_SIMPLEX,           # Font type.
                3,                                   # Font scale (size).
                (0, 255, 255),                       # Text color in BGR (Yellow-Cyan).
                5,                                   # Thickness of the text.
                cv2.LINE_AA                          # Anti-aliased line type.
            )

        return True  # Effect remains active.

# -------------------------------------
# Effects Manager Class
# -------------------------------------

class EffectsManager:
    """
    Manages and orchestrates all active visual effects.
    """
    def __init__(self):
        """
        Initializes the EffectsManager with empty lists for each effect type.
        """
        self.fireworks = []      # List to manage active Firework instances.
        self.neon_effects = []   # List to manage active NeonEffect instances.
        self.clap_effects = []   # List to manage active ClapEffect instances.

    def trigger_firework(self, position):
        """
        Initiates a new firework effect at the specified position.

        Args:
            position: A tuple (x, y) representing where the firework should originate.
        """
        self.fireworks.append(Firework(position))

    def trigger_neon(self, position):
        """
        Initiates a new neon effect at the specified position.

        Args:
            position: A tuple (x, y) representing the center of the neon effect.
        """
        self.neon_effects.append(NeonEffect(position))

    def trigger_clap(self, position):
        """
        Initiates a new clap effect at the specified position.

        Args:
            position: A tuple (x, y) representing where the "WoW" text should appear.
        """
        self.clap_effects.append(ClapEffect(position))

    def update(self, frame):
        """
        Updates all active effects and renders them on the frame.

        Args:
            frame: The current video frame to draw effects on.
        """
        # ---------------------------------
        # Update Fireworks
        # ---------------------------------
        for firework in self.fireworks[:]:  # Iterate over a copy for safe removal.
            firework.update(frame)          # Update and render the firework.
            if not firework.particles:      # If no particles remain, remove the firework.
                self.fireworks.remove(firework)

        # ---------------------------------
        # Update Neon Effects
        # ---------------------------------
        for neon in self.neon_effects[:]:
            active = neon.update(frame)     # Update and render the neon effect.
            if not active:                  # If the effect has ended, remove it.
                self.neon_effects.remove(neon)

        # ---------------------------------
        # Update Clap Effects
        # ---------------------------------
        for clap in self.clap_effects[:]:
            active = clap.update(frame)     # Update and render the clap effect.
            if not active:                  # If the effect has ended, remove it.
                self.clap_effects.remove(clap)

# -------------------------------------
# Show Effect Function
# -------------------------------------

def show_effect(frame, gesture, effects_manager, hand_landmarks):
    """
    Triggers the appropriate visual effect based on the detected gesture.

    Args:
        frame: The current video frame.
        gesture: The detected gesture.
        effects_manager: An instance of EffectsManager to manage effects.
        hand_landmarks: The landmarks of the detected hand.
    """
    # ---------------------------------
    # Trigger Effect Based on Gesture Type
    # ---------------------------------
    if gesture == Gesture.V:
        # ---------------------------------
        # V Gesture: Trigger Firework Effect
        # ---------------------------------
        # Get the wrist landmark to determine the position for the firework.
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        h, w, _ = frame.shape
        position = (int(wrist.x * w), int(wrist.y * h))  # Convert normalized coordinates to pixel values.
        effects_manager.trigger_firework(position)

    elif gesture == Gesture.OK:
        # ---------------------------------
        # OK Gesture: Trigger Neon Effect
        # ---------------------------------
        # Get the wrist landmark to determine the center for the neon effect.
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        h, w, _ = frame.shape
        position = (int(wrist.x * w), int(wrist.y * h))  # Convert normalized coordinates to pixel values.
        effects_manager.trigger_neon(position)

    elif gesture == Gesture.THUMBS_UP:
        # ---------------------------------
        # Thumbs Up Gesture: Trigger Clap Effect
        # ---------------------------------
        # Get the wrist landmark to determine the position for the clap effect.
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        h, w, _ = frame.shape
        position = (int(wrist.x * w), int(wrist.y * h))  # Convert normalized coordinates to pixel values.
        effects_manager.trigger_clap(position)

# -------------------------------------
# Main Function
# -------------------------------------

def main():
    """
    The main execution loop that captures video, detects gestures, and manages visual effects.
    """
    # ---------------------------------
    # Initialize Video Capture
    # ---------------------------------
    cap = cv2.VideoCapture(0)  # Open the default webcam (index 0).
    prev_time = 0             # Variable to calculate frames per second (FPS).
    gesture = Gesture.UNKNOWN # Initialize the current gesture as UNKNOWN.
    effects_manager = EffectsManager()  # Instantiate the EffectsManager.

    # ---------------------------------
    # Main Loop: Process Each Video Frame
    # ---------------------------------
    while True:
        success, frame = cap.read()  # Capture a frame from the webcam.
        if not success:
            break  # If frame capture fails, exit the loop.

        # ---------------------------------
        # Preprocess the Frame
        # ---------------------------------
        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally for a better user experience.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB.

        # ---------------------------------
        # Hand Gesture Detection
        # ---------------------------------
        result = hands.process(img_rgb)  # Process the RGB frame to detect hands.

        # ---------------------------------
        # If Hand Landmarks are Detected
        # ---------------------------------
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # ---------------------------------
                # Detect the Current Gesture
                # ---------------------------------
                current_gesture = detect_gesture(hand_landmarks)

                if current_gesture != Gesture.UNKNOWN:
                    # ---------------------------------
                    # If a New Gesture is Detected, Trigger the Corresponding Effect
                    # ---------------------------------
                    if current_gesture != gesture:
                        gesture = current_gesture  # Update the current gesture.
                        show_effect(frame, gesture, effects_manager, hand_landmarks)  # Trigger the effect.

                    # ---------------------------------
                    # Draw Hand Landmarks and Connections with Custom Styling
                    # ---------------------------------
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=4),  # Green connections with thickness 4.
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=6)   # Red landmarks with thickness 6.
                    )
                else:
                    # ---------------------------------
                    # If No Recognized Gesture, Reset to UNKNOWN
                    # ---------------------------------
                    gesture = Gesture.UNKNOWN

                    # ---------------------------------
                    # Draw Hand Landmarks and Connections with Default Styling
                    # ---------------------------------
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),  # White connections with thickness 2.
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)   # White landmarks with thickness 2.
                    )

        # ---------------------------------
        # Update and Render All Active Effects
        # ---------------------------------
        effects_manager.update(frame)

        # ---------------------------------
        # Calculate and Display Frames Per Second (FPS)
        # ---------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0  # Prevent division by zero.
        prev_time = curr_time  # Update the previous time for the next frame.

        # Overlay the FPS on the frame.
        cv2.putText(
            frame,
            f'FPS: {int(fps)}',              # Text to display.
            (10, 70),                        # Position (x, y) in pixels.
            cv2.FONT_HERSHEY_SIMPLEX,        # Font type.
            1,                               # Font scale (size).
            (255, 0, 0),                     # Text color in BGR (Blue).
            2                                # Thickness of the text.
        )

        # ---------------------------------
        # Display the Processed Frame
        # ---------------------------------
        cv2.imshow("Hand Gesture Recognition", frame)  # Window title.

        # ---------------------------------
        # Exit Condition: Press 'q' to Quit
        # ---------------------------------
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed.

    # ---------------------------------
    # Release Resources and Close Windows
    # ---------------------------------
    cap.release()           # Release the webcam resource.
    cv2.destroyAllWindows()  # Close all OpenCV windows.

# -------------------------------------
# Entry Point of the Script
# -------------------------------------

if __name__ == "__main__":
    main()  # Invoke the main function when the script is executed.
