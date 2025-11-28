import cv2
import mediapipe as mp
import socket
import time
import math
import numpy as np
import pygame
import sys
from threading import Thread
import requests
import json


# -----------------------------
# SMS CONFIGURATION - IMPROVED
# -----------------------------
def send_sms(api_key: str, numbers: str, message: str):
    """Send SMS using Fast2SMS API with better error handling"""
    url = "https://www.fast2sms.com/dev/bulkV2"

    headers = {
        "authorization": api_key,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "route": "q",  # quick SMS route
        "message": message,
        "language": "english",
        "numbers": numbers
    }

    try:
        print(f"Attempting to send SMS to {numbers}...")
        print(f"Message: {message}")

        response = requests.post(url, data=data, headers=headers)
        resp_json = response.json()

        print(f"Fast2SMS API Response: {json.dumps(resp_json, indent=2)}")

        if resp_json.get('return'):
            print(f"✓ SMS Sent Successfully: {message}")
            return True
        else:
            print(f"✗ SMS Failed: {resp_json.get('message', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"✗ SMS sending error: {e}")
        return False


def test_sms():
    """Test SMS functionality independently"""
    print("\n" + "=" * 50)
    print("TESTING SMS FUNCTIONALITY")
    print("=" * 50)

    api_key = "SnjyV7EB8q2Rtm1GJoZQwkv45DKI0cple6gMNWTXdsCfFAriLuoaFzGqCUSb2RWTMAL15n7gpshmIwik"
    numbers = "917019370603"
    test_message = "TEST: Driver Monitoring System SMS Test - System is working!"

    result = send_sms(api_key, numbers, test_message)

    if result:
        print("✓ SMS Test: PASSED")
    else:
        print("✗ SMS Test: FAILED - Check API key and number format")

    print("=" * 50 + "\n")
    return result


# SMS Configuration
SMS_API_KEY = "SnjyV7EB8q2Rtm1GJoZQwkv45DKI0cple6gMNWTXdsCfFAriLuoaFzGqCUSb2RWTMAL15n7gpshmIwik"
SMS_NUMBERS = "919481880458"  # your target number(s)
SMS_EMERGENCY_THRESHOLD = 10  # Send SMS after 10 seconds of sleep
sms_sent = False  # Flag to track if SMS has been sent
sms_test_completed = False  # Track if initial test was done

# -----------------------------
# UDP CONFIG FOR NODEMCU ACCESS POINT
# -----------------------------
NODEMCU_AP_IP = "10.155.87.136"  # NodeMCU Access Point IP
NODEMCU_AP_PORT = 4210
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

last_sent = None
last_time = 0
DEBOUNCE_SECONDS = 0.15


def send_udp(msg: str):
    """Send driver state to NodeMCU Access Point"""
    global last_sent, last_time
    now = time.time()

    if msg == last_sent and (now - last_time) < DEBOUNCE_SECONDS:
        return

    last_sent = msg
    last_time = now
    try:
        sock.sendto(msg.encode("utf-8"), (NODEMCU_AP_IP, NODEMCU_AP_PORT))
        print(f"-> NodeMCU AP: {msg}")
    except Exception as e:
        print(f"UDP send error: {e}")


# -----------------------------
# CAR SIMULATION SETUP
# -----------------------------
pygame.init()

# Screen dimensions
CAR_SCREEN_WIDTH = 800
CAR_SCREEN_HEIGHT = 600
ROAD_WIDTH = 400

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GRAY = (50, 50, 50)
AQUA = (0, 255, 255)

# Car properties
car_width = 60
car_height = 100
car_x = CAR_SCREEN_WIDTH // 2 - car_width // 2
car_y = CAR_SCREEN_HEIGHT - 150
car_speed = 5
max_speed = 8
acceleration = 0.1

# Road properties
road_x = (CAR_SCREEN_WIDTH - ROAD_WIDTH) // 2
lane_width = ROAD_WIDTH // 3
left_lane = road_x + lane_width - car_width // 2
center_lane = road_x + 2 * lane_width - car_width // 2
right_lane = road_x + 3 * lane_width - car_width // 2

current_lane = center_lane
target_x = current_lane

# Road movement simulation
road_scroll = 0
road_scroll_speed = 3

# Car simulation state
car_running = False
car_screen = None
emergency_triggered_car = False
sleep_start_time_car = None
emergency_maneuver_start_time = None
recovery_start_time = None
SLEEP_EMERGENCY_THRESHOLD_CAR = 3
MANEUVER_DURATION = 4
RECOVERY_DURATION = 5

# Indicator variables
LEFT_TURN_INDICATOR_INTERVAL = 0.2  # Rapid blinking for left turns

# Track emergency completion
emergency_completed = False
in_emergency_sequence = False

# Track if driver has ever slept
driver_has_slept = False


def init_car_simulation():
    """Initialize the car simulation window"""
    global car_screen
    car_screen = pygame.display.set_mode((CAR_SCREEN_WIDTH, CAR_SCREEN_HEIGHT))
    pygame.display.set_caption("Car Simulation - Driver Monitoring")


def draw_road():
    """Draw the road with moving lanes"""
    global road_scroll

    # Road background
    pygame.draw.rect(car_screen, DARK_GRAY, (road_x, 0, ROAD_WIDTH, CAR_SCREEN_HEIGHT))

    # Lane markings with scrolling effect
    for i in range(1, 3):
        lane_x = road_x + i * lane_width
        for y in range(-40, CAR_SCREEN_HEIGHT, 40):
            pygame.draw.rect(car_screen, WHITE, (lane_x - 2, (y + road_scroll) % CAR_SCREEN_HEIGHT, 4, 20))

    # Road edges
    pygame.draw.rect(car_screen, WHITE, (road_x, 0, 5, CAR_SCREEN_HEIGHT))
    pygame.draw.rect(car_screen, WHITE, (road_x + ROAD_WIDTH - 5, 0, 5, CAR_SCREEN_HEIGHT))

    # Update road scroll for moving effect
    if car_speed > 0:
        road_scroll = (road_scroll + road_scroll_speed * (car_speed / max_speed)) % 40


def draw_car():
    """Draw the car at its current position"""
    # Car body
    pygame.draw.rect(car_screen, AQUA, (car_x, car_y, car_width, car_height))

    # Windows
    pygame.draw.rect(car_screen, BLUE, (car_x + 5, car_y + 5, car_width - 10, 30))
    pygame.draw.rect(car_screen, BLUE, (car_x + 5, car_y + 40, car_width - 10, 30))

    # Wheels
    pygame.draw.rect(car_screen, BLACK, (car_x - 5, car_y + 10, 5, 20))
    pygame.draw.rect(car_screen, BLACK, (car_x + car_width, car_y + 10, 5, 20))
    pygame.draw.rect(car_screen, BLACK, (car_x - 5, car_y + 70, 5, 20))
    pygame.draw.rect(car_screen, BLACK, (car_x + car_width, car_y + 70, 5, 20))

    current_time = time.time()

    # Check if we're in emergency maneuver and moving to left lane
    is_moving_to_left_lane = (emergency_triggered_car and
                              emergency_maneuver_start_time and
                              (current_time - emergency_maneuver_start_time) <= MANEUVER_DURATION * 0.5)

    # Check if we're in emergency stopping phase in left lane
    is_emergency_stopping = (emergency_triggered_car and
                             emergency_maneuver_start_time and
                             (current_time - emergency_maneuver_start_time) > MANEUVER_DURATION * 0.5)

    # Determine indicator state
    if is_moving_to_left_lane or (is_emergency_stopping and target_x == left_lane):
        # Blinking indicators during lane change and stopping
        left_turn_indicator_on = (current_time % LEFT_TURN_INDICATOR_INTERVAL) < (LEFT_TURN_INDICATOR_INTERVAL / 2)
        if left_turn_indicator_on:
            # Blinking yellow headlights and red rear indicators
            front_headlight_color = YELLOW
            rear_indicator_color = RED
        else:
            # Turn off indicators when not blinking
            front_headlight_color = AQUA
            rear_indicator_color = AQUA
    else:
        # Normal yellow headlights and red rear indicators when not turning/stopping
        front_headlight_color = YELLOW
        rear_indicator_color = RED

    # Draw front headlights
    pygame.draw.rect(car_screen, front_headlight_color, (car_x + 5, car_y, 10, 5))
    pygame.draw.rect(car_screen, front_headlight_color, (car_x + car_width - 15, car_y, 10, 5))

    # Draw rear indicators
    pygame.draw.rect(car_screen, rear_indicator_color, (car_x + 8, car_y + car_height - 20, 6, 6))
    pygame.draw.rect(car_screen, rear_indicator_color,
                     (car_x + car_width - 14, car_y + car_height - 20, 6, 6))


def draw_dashboard():
    """Draw the dashboard with driver state and car info"""
    global sms_sent

    # Dashboard background
    pygame.draw.rect(car_screen, BLACK, (0, 0, CAR_SCREEN_WIDTH, 80))

    # Driver state
    state_text = f"Driver State: {current_driver_state}"
    if in_emergency_sequence:
        state_color = ORANGE
    elif emergency_completed:
        state_color = GREEN
    else:
        state_color = GREEN if current_driver_state != "DRIVER_SLEPT" else YELLOW if not emergency_triggered_car else RED

    font = pygame.font.SysFont('Arial', 24)
    state_surface = font.render(state_text, True, state_color)
    car_screen.blit(state_surface, (20, 20))

    # Speed and lane info
    speed_text = f"Speed: {car_speed:.1f} km/h"
    speed_surface = font.render(speed_text, True, WHITE)
    car_screen.blit(speed_surface, (20, 50))

    lane_text = f"Lane: {'LEFT' if target_x == left_lane else 'CENTER' if target_x == center_lane else 'RIGHT'}"
    lane_surface = font.render(lane_text, True, WHITE)
    car_screen.blit(lane_surface, (250, 50))

    # NodeMCU Connection status
    connection_text = f"NodeMCU: Connected to AP"
    connection_surface = font.render(connection_text, True, GREEN)
    car_screen.blit(connection_surface, (450, 50))

    # SMS Status
    sms_status = f"SMS: {'SENT' if sms_sent else 'READY'}"
    sms_color = RED if sms_sent else GREEN
    sms_surface = font.render(sms_status, True, sms_color)
    car_screen.blit(sms_surface, (600, 20))

    # Left turn indicator status
    current_time = time.time()
    is_moving_to_left_lane = (emergency_triggered_car and
                              emergency_maneuver_start_time and
                              (current_time - emergency_maneuver_start_time) <= MANEUVER_DURATION * 0.5)

    is_emergency_stopping = (emergency_triggered_car and
                             emergency_maneuver_start_time and
                             (current_time - emergency_maneuver_start_time) > MANEUVER_DURATION * 0.5)

    is_turning_left = is_moving_to_left_lane or (is_emergency_stopping and target_x == left_lane)

    left_indicator_text = f"Left Turn: {'ON' if is_turning_left else 'OFF'}"
    left_indicator_surface = font.render(left_indicator_text, True, YELLOW if is_turning_left else WHITE)
    car_screen.blit(left_indicator_surface, (600, 50))

    # Status messages
    if emergency_triggered_car:
        if emergency_maneuver_start_time:
            maneuver_progress = (time.time() - emergency_maneuver_start_time) / MANEUVER_DURATION
            if maneuver_progress < 0.5:
                status_text = "EMERGENCY! MOVING TO LEFT LANE WITH INDICATORS..."
            elif maneuver_progress < 1.0:
                status_text = "EMERGENCY! STOPPING IN LEFT LANE WITH INDICATORS..."
            else:
                status_text = "EMERGENCY! CAR STOPPED!"
        else:
            status_text = "EMERGENCY! MOVING TO LEFT LANE WITH INDICATORS..."
    elif current_driver_state == "DRIVER_SLEPT":
        if sleep_start_time_car:
            sleep_duration = time.time() - sleep_start_time_car
            remaining_time = max(0, SLEEP_EMERGENCY_THRESHOLD_CAR - sleep_duration)
            status_text = f"DRIVER ASLEEP! Emergency in {remaining_time:.1f}s"

            # Check for SMS sending condition
            if sleep_duration >= SMS_EMERGENCY_THRESHOLD and not sms_sent:
                status_text += " - SMS SENT!"
        else:
            status_text = "DRIVER ASLEEP!"
    elif in_emergency_sequence:
        if recovery_start_time:
            wait_elapsed = time.time() - recovery_start_time
            wait_remaining = max(0, RECOVERY_DURATION - wait_elapsed)
            if wait_remaining > 0:
                status_text = f"WAITING: {wait_remaining:.1f}s before moving..."
            else:
                status_text = "CONTINUING IN LEFT LANE"
        else:
            status_text = "POST-EMERGENCY SEQUENCE ACTIVE"
    elif emergency_completed:
        status_text = "DRIVING IN LEFT LANE (POST-EMERGENCY)"
    else:
        status_text = "NORMAL DRIVING"

    # Add indicator if driver has slept
    if driver_has_slept:
        status_text += " - DRIVER HAS SLEPT ONCE"

    status_surface = font.render(status_text, True, state_color)
    car_screen.blit(status_surface, (400, 20))


def update_car():
    """Update car position and speed based on driver state"""
    global car_x, car_speed, target_x, emergency_triggered_car, sleep_start_time_car
    global emergency_maneuver_start_time, recovery_start_time
    global emergency_completed, in_emergency_sequence, driver_has_slept

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    # Start emergency sequence when driver sleeps for the first time
    if not in_emergency_sequence and current_driver_state == "DRIVER_SLEPT" and not driver_has_slept:
        in_emergency_sequence = True
        sleep_start_time_car = time.time()
        driver_has_slept = True
        print("Starting emergency sequence...")

    if in_emergency_sequence:
        if current_driver_state == "DRIVER_SLEPT":
            # Driver is still asleep - continue emergency
            sleep_duration = time.time() - sleep_start_time_car

            if sleep_duration < SLEEP_EMERGENCY_THRESHOLD_CAR:
                # Countdown phase - maintain normal driving
                if car_speed < max_speed:
                    car_speed += acceleration
                target_x = center_lane
            else:
                # Emergency maneuver phase
                if not emergency_triggered_car:
                    emergency_triggered_car = True
                    emergency_maneuver_start_time = time.time()
                    print("EMERGENCY: Starting safety maneuver with indicators!")

                if emergency_maneuver_start_time:
                    maneuver_progress = min(1.0, (time.time() - emergency_maneuver_start_time) / MANEUVER_DURATION)

                    if maneuver_progress < 0.5:
                        # Move to left lane with indicators blinking
                        lane_progress = maneuver_progress / 0.5
                        target_x = center_lane + (left_lane - center_lane) * lane_progress
                        if car_speed < max_speed:
                            car_speed += acceleration
                    else:
                        # Stop the car in left lane with indicators blinking
                        target_x = left_lane
                        brake_progress = (maneuver_progress - 0.5) / 0.5
                        car_speed = max_speed * (1 - brake_progress)
                        if car_speed < 0:
                            car_speed = 0
        else:
            # Driver woke up - start recovery
            if emergency_triggered_car and car_speed == 0 and recovery_start_time is None:
                recovery_start_time = time.time()
                print("Driver woke up! Starting 5-second wait...")

            if recovery_start_time is not None:
                wait_elapsed = time.time() - recovery_start_time

                if wait_elapsed < RECOVERY_DURATION:
                    # Wait period
                    car_speed = 0
                    target_x = left_lane
                else:
                    # Recovery driving
                    recovery_driving_elapsed = wait_elapsed - RECOVERY_DURATION
                    recovery_progress = min(1.0, recovery_driving_elapsed / 5.0)

                    # Accelerate slowly
                    target_speed = max_speed * 0.5 * recovery_progress
                    if car_speed < target_speed:
                        car_speed += acceleration * 0.3

                    # Always stay in left lane after driver has slept
                    target_x = left_lane

                    if recovery_progress >= 1.0:
                        # Recovery complete
                        emergency_triggered_car = False
                        emergency_completed = True
                        in_emergency_sequence = False
                        recovery_start_time = None
                        sleep_start_time_car = None
                        emergency_maneuver_start_time = None
                        print("Recovery complete! Continuing in left lane.")
    else:
        # Normal driving (no emergency sequence active)
        if car_speed < max_speed:
            car_speed += acceleration

        # If driver has slept at least once, always stay in left lane after emergency completion
        if driver_has_slept and emergency_completed:
            target_x = left_lane
        else:
            target_x = center_lane

    # Smooth lane changing
    if abs(car_x - target_x) > 1:
        lane_change_speed = 2 if car_speed > 0 else 1
        if car_x < target_x:
            car_x += lane_change_speed
        else:
            car_x -= lane_change_speed

    return True


def run_car_simulation():
    """Main loop for car simulation"""
    global car_running

    init_car_simulation()
    clock = pygame.time.Clock()
    car_running = True

    while car_running:
        # Update car based on driver state
        if not update_car():
            break

        # Draw everything
        car_screen.fill(WHITE)
        draw_road()
        draw_car()
        draw_dashboard()

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# -----------------------------
# FACE DETECTION SETUP
# -----------------------------
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Correct MediaPipe eye-landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Facial landmarks
CHIN = 152
LEFT_SHOULDER = 234
RIGHT_SHOULDER = 454

# Head pose landmarks
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

# Thresholds
EYE_CLOSED_THRESHOLD = 0.23

# Sleep detection variables
sleep_start_time = None
SLEEP_EMERGENCY_THRESHOLD = 3
emergency_triggered = False

# Driver state for car simulation
current_driver_state = "NORMAL"


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def eye_aspect_ratio(landmarks, eye):
    """Compute EAR for given eye using MediaPipe normalized coordinates"""
    v1 = distance(landmarks[eye[1]], landmarks[eye[5]])
    v2 = distance(landmarks[eye[2]], landmarks[eye[4]])
    h = distance(landmarks[eye[0]], landmarks[eye[3]])
    return (v1 + v2) / (2.0 * h)


def classify_eye_state(left_ear, right_ear):
    left_closed = left_ear < EYE_CLOSED_THRESHOLD
    right_closed = right_ear < EYE_CLOSED_THRESHOLD

    if left_closed and right_closed:
        return "BOTH_CLOSED"
    elif left_closed:
        return "LEFT_CLOSED"
    elif right_closed:
        return "RIGHT_CLOSED"
    else:
        return "OPEN"


def chin_below_shoulders(landmarks):
    """Check if the chin is lower than both shoulders"""
    chin_y = landmarks[CHIN].y
    left_shoulder_y = landmarks[LEFT_SHOULDER].y
    right_shoulder_y = landmarks[RIGHT_SHOULDER].y

    return chin_y > left_shoulder_y and chin_y > right_shoulder_y


def get_head_pose(landmarks, image_shape):
    """Determine head pose using solvePnP approach"""
    img_h, img_w, img_c = image_shape
    face_2d = []
    face_3d = []

    for idx in HEAD_POSE_LANDMARKS:
        lm = landmarks[idx]
        if idx == 1:  # Nose tip
            nose_2d = (lm.x * img_w, lm.y * img_h)
            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])

    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, distortion_matrix
    )

    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Correct head direction logic
    if y < -10:
        head_direction = "LOOKING_LEFT"
    elif y > 10:
        head_direction = "LOOKING_RIGHT"
    elif x > 10:
        head_direction = "LOOKING_DOWN"
    elif x < -10:
        head_direction = "LOOKING_UP"
    else:
        head_direction = "FORWARD"

    return head_direction, x, y, z, nose_2d, nose_3d


# -----------------------------
# MAIN LOOP - IMPROVED SMS HANDLING
# -----------------------------
def main():
    global sleep_start_time, emergency_triggered, current_driver_state, car_running, sms_sent, sms_test_completed

    print("Starting Driver Monitoring System with NodeMCU AP Connection and SMS")
    print(f"NodeMCU AP: {NODEMCU_AP_IP}:{NODEMCU_AP_PORT}")
    print(f"SMS Emergency Threshold: {SMS_EMERGENCY_THRESHOLD} seconds")
    print("Make sure your computer is connected to the NodeMCU Access Point!")

    # Test SMS functionality at startup
    if not sms_test_completed:
        sms_test_completed = test_sms()

    # Start car simulation in a separate thread
    car_thread = Thread(target=run_car_simulation, daemon=True)
    car_thread.start()

    # Wait for car simulation to initialize
    time.sleep(2)

    cap = cv2.VideoCapture(0)

    # Initialize FPS calculation
    start_time = time.time()
    frame_count = 0
    fps = 0

    with mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as face:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = end_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Eye state detection
                leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
                rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
                eye_state = classify_eye_state(leftEAR, rightEAR)

                # Head pose detection
                head_direction, x_angle, y_angle, z_angle, nose_2d, nose_3d = get_head_pose(landmarks, frame.shape)

                # Combined sleep detection
                slept_detected = False
                if eye_state == "BOTH_CLOSED" and (head_direction == "LOOKING_DOWN" or chin_below_shoulders(landmarks)):
                    slept_detected = True
                    combined_state = "DRIVER_SLEPT"
                    current_driver_state = "DRIVER_SLEPT"

                    if sleep_start_time is None:
                        sleep_start_time = time.time()
                        sms_sent = False  # Reset SMS flag when sleep starts
                        print(f"Sleep detected at {time.strftime('%H:%M:%S')}")

                    sleep_duration = time.time() - sleep_start_time

                    # Check if we need to send SMS
                    if sleep_duration >= SMS_EMERGENCY_THRESHOLD and not sms_sent:
                        print(f"⏰ SMS Trigger: Sleep duration {sleep_duration:.1f}s >= {SMS_EMERGENCY_THRESHOLD}s")
                        sms_message = f"🚨 EMERGENCY ALERT: Driver asleep for {sleep_duration:.1f}s! Vehicle has initiated safety protocol. Time: {time.strftime('%H:%M:%S')}"
                        sms_result = send_sms(SMS_API_KEY, SMS_NUMBERS, sms_message)
                        sms_sent = sms_result  # Only mark as sent if successful
                        if sms_result:
                            print(f"✓ Emergency SMS sent after {sleep_duration:.1f} seconds of sleep")
                        else:
                            print(f"✗ Failed to send emergency SMS")

                    if sleep_duration >= SLEEP_EMERGENCY_THRESHOLD and not emergency_triggered:
                        emergency_triggered = True
                        print("EMERGENCY: Driver slept for more than 3 seconds!")

                else:
                    combined_state = "NORMAL"
                    current_driver_state = "NORMAL"
                    if sleep_start_time is not None:
                        # Only reset if driver was previously sleeping
                        sleep_duration = time.time() - sleep_start_time
                        if sleep_duration > 1:  # Only log if sleep was significant
                            print(f"Driver woke up after {sleep_duration:.1f}s sleep")
                        sleep_start_time = None

                    if emergency_triggered:
                        emergency_triggered = False
                        print("Emergency cleared - driver awake")

                # Send simplified state to NodeMCU
                send_udp(combined_state)

                # Visual overlays
                cv2.putText(frame, f"L_EAR: {leftEAR:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"R_EAR: {rightEAR:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"EYES: {eye_state}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
                cv2.putText(frame, f"HEAD: {head_direction}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(frame, f"X: {x_angle:.1f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Y: {y_angle:.1f}", (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Z: {z_angle:.1f}", (190, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                status_color = (0, 0, 255) if slept_detected else (255, 255, 255)
                cv2.putText(frame, f"STATE: {combined_state}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                if slept_detected and sleep_start_time is not None:
                    sleep_duration = time.time() - sleep_start_time
                    cv2.putText(frame, f"SLEEP TIMER: {sleep_duration:.1f}s", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Show SMS status
                    if sleep_duration >= SMS_EMERGENCY_THRESHOLD:
                        sms_status = "SMS: SENT" if sms_sent else "SMS: PENDING"
                        sms_color = (0, 255, 0) if sms_sent else (0, 255, 255)
                        cv2.putText(frame, sms_status, (10, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sms_color, 2)

                    if emergency_triggered:
                        cv2.putText(frame, "EMERGENCY! SAFETY MANEUVER ACTIVATED", (10, 270),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        remaining_time = max(0, SLEEP_EMERGENCY_THRESHOLD - sleep_duration)
                        cv2.putText(frame, f"DRIVER ASLEEP! Emergency in {remaining_time:.1f}s", (10, 270),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw head pose direction line
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
                cv2.line(frame, p1, p2, (255, 0, 0), 3)

                # Draw face landmarks
                mp_draw.draw_landmarks(
                    frame,
                    results.multi_face_landmarks[0],
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

            else:
                send_udp("NO_FACE")
                current_driver_state = "NO_FACE"
                cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                sleep_start_time = None
                if emergency_triggered:
                    emergency_triggered = False

            # Display FPS
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Driver Monitoring System", frame)

            # Check for ESC key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                car_running = False
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    car_running = False
    pygame.quit()


if __name__ == "__main__":
    main()