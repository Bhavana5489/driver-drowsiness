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

    # Use the same number for testing as for emergencies
    api_key = "YOUR_API_KEY"
    numbers = "911234567890"  # Your number
    test_message = "TEST: Driver Monitoring System SMS Test - System is working!"

    result = send_sms(api_key, numbers, test_message)

    if result:
        print("✓ SMS Test: PASSED")
    else:
        print("✗ SMS Test: FAILED - Check API key and number format")

    print("=" * 50 + "\n")
    return result


# SMS Configuration - USING YOUR NUMBER FOR BOTH TEST AND EMERGENCY
SMS_API_KEY = "YOUR_API_KEY"
SMS_NUMBERS = "911234567890"  # CHANGED TO YOUR NUMBER
SMS_EMERGENCY_THRESHOLD = 10  # Send SMS after 10 seconds of sleep
SMS_NO_FACE_THRESHOLD = 10  # Send SMS after 10 seconds of no face detection
sms_sent = False  # Flag to track if SMS has been sent for sleep
no_face_sms_sent = False  # Flag to track if SMS has been sent for no face
sms_test_completed = True  # Set to True to skip automatic test

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
# CAR SIMULATION SETUP - ENHANCED
# -----------------------------
pygame.init()

# Screen dimensions
CAR_SCREEN_WIDTH = 900
CAR_SCREEN_HEIGHT = 700
ROAD_WIDTH = 500

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
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)
BROWN = (139, 69, 19)
LIGHT_GRAY = (200, 200, 200)

# Car properties - Enhanced
car_width = 80
car_height = 160
car_x = CAR_SCREEN_WIDTH // 2 - car_width // 2
car_y = CAR_SCREEN_HEIGHT - 200
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

# No face emergency variables
no_face_emergency_triggered = False
no_face_emergency_start_time = None
NO_FACE_EMERGENCY_THRESHOLD = 10  # Stop car after 10 seconds of no face
NO_FACE_RECOVERY_DURATION = 5  # Wait 5 seconds after face returns

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
    """Draw the road with moving lanes - Enhanced"""
    global road_scroll

    # Sky background
    pygame.draw.rect(car_screen, LIGHT_BLUE, (0, 0, CAR_SCREEN_WIDTH, CAR_SCREEN_HEIGHT))

    # Road background with gradient
    for i in range(ROAD_WIDTH):
        shade = max(30, 100 - i // 10)
        pygame.draw.line(car_screen, (shade, shade, shade),
                         (road_x + i, 0), (road_x + i, CAR_SCREEN_HEIGHT))

    # Lane markings with scrolling effect
    for i in range(1, 3):
        lane_x = road_x + i * lane_width
        for y in range(-40, CAR_SCREEN_HEIGHT, 40):
            pygame.draw.rect(car_screen, YELLOW, (lane_x - 2, (y + road_scroll) % CAR_SCREEN_HEIGHT, 4, 25))

    # Road edges with rumble strips
    for y in range(-20, CAR_SCREEN_HEIGHT, 40):
        pygame.draw.rect(car_screen, WHITE, (road_x, (y + road_scroll) % CAR_SCREEN_HEIGHT, 8, 20))
        pygame.draw.rect(car_screen, WHITE, (road_x + ROAD_WIDTH - 8, (y + road_scroll) % CAR_SCREEN_HEIGHT, 8, 20))

    # Update road scroll for moving effect
    if car_speed > 0:
        road_scroll = (road_scroll + road_scroll_speed * (car_speed / max_speed)) % 40


def draw_car():
    """Draw the car at its current position - Enhanced"""
    # Car body with gradient and 3D effect
    pygame.draw.rect(car_screen, DARK_BLUE, (car_x, car_y, car_width, car_height), border_radius=10)
    pygame.draw.rect(car_screen, BLUE, (car_x + 5, car_y + 5, car_width - 10, car_height - 10), border_radius=8)

    # Windows with gradient
    pygame.draw.rect(car_screen, LIGHT_BLUE, (car_x + 10, car_y + 15, car_width - 20, 25), border_radius=5)
    pygame.draw.rect(car_screen, LIGHT_BLUE, (car_x + 10, car_y + 50, car_width - 20, 40), border_radius=5)

    # Headlights and details
    pygame.draw.ellipse(car_screen, LIGHT_GRAY, (car_x + 5, car_y + 10, 15, 8))
    pygame.draw.ellipse(car_screen, LIGHT_GRAY, (car_x + car_width - 20, car_y + 10, 15, 8))

    # Side details
    pygame.draw.rect(car_screen, LIGHT_GRAY, (car_x + 5, car_y + 40, car_width - 10, 5))

    # Wheels with 3D effect
    pygame.draw.ellipse(car_screen, BLACK, (car_x - 8, car_y + 20, 16, 30))
    pygame.draw.ellipse(car_screen, BLACK, (car_x + car_width - 8, car_y + 20, 16, 30))
    pygame.draw.ellipse(car_screen, BLACK, (car_x - 8, car_y + car_height - 50, 16, 30))
    pygame.draw.ellipse(car_screen, BLACK, (car_x + car_width - 8, car_y + car_height - 50, 16, 30))

    # Wheel hubs
    pygame.draw.ellipse(car_screen, LIGHT_GRAY, (car_x - 4, car_y + 30, 8, 10))
    pygame.draw.ellipse(car_screen, LIGHT_GRAY, (car_x + car_width - 4, car_y + 30, 8, 10))
    pygame.draw.ellipse(car_screen, LIGHT_GRAY, (car_x - 4, car_y + car_height - 40, 8, 10))
    pygame.draw.ellipse(car_screen, LIGHT_GRAY, (car_x + car_width - 4, car_y + car_height - 40, 8, 10))

    current_time = time.time()

    # Check if we're in emergency maneuver and moving to left lane
    is_moving_to_left_lane = (emergency_triggered_car and
                              emergency_maneuver_start_time and
                              (current_time - emergency_maneuver_start_time) <= MANEUVER_DURATION * 0.5)

    # Check if we're in emergency stopping phase in left lane
    is_emergency_stopping = (emergency_triggered_car and
                             emergency_maneuver_start_time and
                             (current_time - emergency_maneuver_start_time) > MANEUVER_DURATION * 0.5)

    # Check if we're in no face emergency
    is_no_face_emergency = no_face_emergency_triggered

    # Determine indicator state
    if is_moving_to_left_lane or (is_emergency_stopping and target_x == left_lane) or is_no_face_emergency:
        # Blinking indicators during lane change, stopping, or no face emergency
        left_turn_indicator_on = (current_time % LEFT_TURN_INDICATOR_INTERVAL) < (LEFT_TURN_INDICATOR_INTERVAL / 2)
        if left_turn_indicator_on:
            # Blinking yellow indicators
            indicator_color = YELLOW
        else:
            # Turn off indicators when not blinking
            indicator_color = LIGHT_GRAY
    else:
        # Normal indicators when not turning/stopping
        indicator_color = RED

    # Draw front indicators
    pygame.draw.ellipse(car_screen, indicator_color, (car_x + 5, car_y, 10, 8))
    pygame.draw.ellipse(car_screen, indicator_color, (car_x + car_width - 15, car_y, 10, 8))

    # Draw rear indicators
    pygame.draw.ellipse(car_screen, indicator_color, (car_x + 8, car_y + car_height - 10, 10, 8))
    pygame.draw.ellipse(car_screen, indicator_color, (car_x + car_width - 18, car_y + car_height - 10, 10, 8))

    # Headlights (always on when driving)
    if car_speed > 0:
        pygame.draw.ellipse(car_screen, WHITE, (car_x + 2, car_y + 5, 12, 6))
        pygame.draw.ellipse(car_screen, WHITE, (car_x + car_width - 14, car_y + 5, 12, 6))


def draw_dashboard():
    """Draw the dashboard with driver state and car info - Enhanced"""
    global sms_sent, no_face_sms_sent

    # Dashboard background with gradient
    for i in range(150):
        shade = min(255, 50 + i)
        pygame.draw.rect(car_screen, (shade, shade, shade), (0, 0, CAR_SCREEN_WIDTH, 150))

    # Dashboard border
    pygame.draw.rect(car_screen, DARK_GRAY, (0, 0, CAR_SCREEN_WIDTH, 150), 3)
    pygame.draw.rect(car_screen, LIGHT_GRAY, (10, 10, CAR_SCREEN_WIDTH - 20, 130), 2, border_radius=10)

    # Driver state with colored background
    state_text = f"Driver State: {current_driver_state}"
    if in_emergency_sequence or no_face_emergency_triggered:
        state_color = ORANGE
        bg_color = (255, 200, 100)
    elif emergency_completed:
        state_color = GREEN
        bg_color = (100, 255, 100)
    else:
        state_color = GREEN if current_driver_state != "DRIVER_SLEPT" else YELLOW if not emergency_triggered_car else RED
        bg_color = (100, 255, 100) if current_driver_state != "DRIVER_SLEPT" else (255, 255,
                                                                                   100) if not emergency_triggered_car else (
            255, 100, 100)

    # State display box
    pygame.draw.rect(car_screen, bg_color, (20, 20, 300, 40), border_radius=8)
    pygame.draw.rect(car_screen, DARK_GRAY, (20, 20, 300, 40), 2, border_radius=8)

    font = pygame.font.SysFont('Arial', 24, bold=True)
    state_surface = font.render(state_text, True, state_color)
    car_screen.blit(state_surface, (35, 30))

    # Speed display with gauge
    speed_text = f"Speed: {car_speed:.1f} km/h"
    pygame.draw.rect(car_screen, LIGHT_GRAY, (20, 70, 200, 40), border_radius=8)
    pygame.draw.rect(car_screen, DARK_GRAY, (20, 70, 200, 40), 2, border_radius=8)

    speed_surface = font.render(speed_text, True, BLUE)
    car_screen.blit(speed_surface, (35, 80))

    # Speed gauge visualization
    gauge_width = 150
    gauge_fill = (car_speed / max_speed) * gauge_width
    pygame.draw.rect(car_screen, DARK_GRAY, (240, 80, gauge_width, 15), border_radius=5)
    pygame.draw.rect(car_screen,
                     GREEN if car_speed < max_speed * 0.7 else ORANGE if car_speed < max_speed * 0.9 else RED,
                     (240, 80, gauge_fill, 15), border_radius=5)

    # Lane info
    lane_text = f"Lane: {'LEFT' if target_x == left_lane else 'CENTER' if target_x == center_lane else 'RIGHT'}"
    lane_surface = font.render(lane_text, True, DARK_BLUE)
    car_screen.blit(lane_surface, (420, 30))

    # NodeMCU Connection status
    connection_text = f"NodeMCU: Connected"
    connection_surface = font.render(connection_text, True, GREEN)
    car_screen.blit(connection_surface, (420, 70))

    # SMS Status
    sms_status = f"SMS: {'SENT' if sms_sent or no_face_sms_sent else 'READY'}"
    sms_color = RED if sms_sent or no_face_sms_sent else GREEN
    sms_surface = font.render(sms_status, True, sms_color)
    car_screen.blit(sms_surface, (650, 30))

    # Left turn indicator status
    current_time = time.time()
    is_moving_to_left_lane = (emergency_triggered_car and
                              emergency_maneuver_start_time and
                              (current_time - emergency_maneuver_start_time) <= MANEUVER_DURATION * 0.5)

    is_emergency_stopping = (emergency_triggered_car and
                             emergency_maneuver_start_time and
                             (current_time - emergency_maneuver_start_time) > MANEUVER_DURATION * 0.5)

    is_turning_left = is_moving_to_left_lane or (
                is_emergency_stopping and target_x == left_lane) or no_face_emergency_triggered

    left_indicator_text = f"Left Indicator: {'BLINKING' if is_turning_left else 'OFF'}"
    left_indicator_color = YELLOW if is_turning_left else LIGHT_GRAY
    left_indicator_surface = font.render(left_indicator_text, True, left_indicator_color)
    car_screen.blit(left_indicator_surface, (650, 70))

    # Status messages panel
    status_panel_y = 110
    pygame.draw.rect(car_screen, LIGHT_GRAY, (20, status_panel_y, CAR_SCREEN_WIDTH - 40, 30), border_radius=5)
    pygame.draw.rect(car_screen, DARK_GRAY, (20, status_panel_y, CAR_SCREEN_WIDTH - 40, 30), 2, border_radius=5)

    if emergency_triggered_car:
        if emergency_maneuver_start_time:
            maneuver_progress = (time.time() - emergency_maneuver_start_time) / MANEUVER_DURATION
            if maneuver_progress < 0.5:
                status_text = "🚨 EMERGENCY! MOVING TO LEFT LANE WITH INDICATORS..."
            elif maneuver_progress < 1.0:
                status_text = "🚨 EMERGENCY! STOPPING IN LEFT LANE WITH INDICATORS..."
            else:
                status_text = "🚨 EMERGENCY! CAR STOPPED!"
        else:
            status_text = "🚨 EMERGENCY! MOVING TO LEFT LANE WITH INDICATORS..."
    elif no_face_emergency_triggered:
        status_text = "🚨 NO FACE EMERGENCY! CAR STOPPED WITH INDICATORS!"
    elif current_driver_state == "DRIVER_SLEPT":
        if sleep_start_time_car:
            sleep_duration = time.time() - sleep_start_time_car
            remaining_time = max(0, SLEEP_EMERGENCY_THRESHOLD_CAR - sleep_duration)
            status_text = f"⚠️ DRIVER ASLEEP! Emergency in {remaining_time:.1f}s"

            # Check for SMS sending condition
            if sleep_duration >= SMS_EMERGENCY_THRESHOLD and not sms_sent:
                status_text += " - 📱 SMS SENT!"
        else:
            status_text = "⚠️ DRIVER ASLEEP!"
    elif current_driver_state == "NO_FACE":
        if no_face_start_time:
            no_face_duration = time.time() - no_face_start_time
            remaining_time = max(0, NO_FACE_EMERGENCY_THRESHOLD - no_face_duration)
            status_text = f"⚠️ NO FACE DETECTED! Emergency stop in {remaining_time:.1f}s"

            # Check for SMS sending condition
            if no_face_duration >= SMS_NO_FACE_THRESHOLD and not no_face_sms_sent:
                status_text += " - 📱 SMS SENT!"
        else:
            status_text = "⚠️ NO FACE DETECTED!"
    elif in_emergency_sequence:
        if recovery_start_time:
            wait_elapsed = time.time() - recovery_start_time
            wait_remaining = max(0, RECOVERY_DURATION - wait_elapsed)
            if wait_remaining > 0:
                status_text = f"⏳ WAITING: {wait_remaining:.1f}s before moving..."
            else:
                status_text = "🔄 CONTINUING IN LEFT LANE"
        else:
            status_text = "🔄 POST-EMERGENCY SEQUENCE ACTIVE"
    elif emergency_completed:
        status_text = "✅ DRIVING IN LEFT LANE (POST-EMERGENCY)"
    else:
        status_text = "✅ NORMAL DRIVING"

    # Add indicator if driver has slept
    if driver_has_slept:
        status_text += " - ⚠️ DRIVER HAS SLEPT ONCE"

    status_font = pygame.font.SysFont('Arial', 20, bold=True)
    status_surface = status_font.render(status_text, True, DARK_GRAY)
    car_screen.blit(status_surface, (30, status_panel_y + 5))


def update_car():
    """Update car position and speed based on driver state - FIXED FOR MULTIPLE SLEEP EVENTS"""
    global car_x, car_speed, target_x, emergency_triggered_car, sleep_start_time_car
    global emergency_maneuver_start_time, recovery_start_time
    global emergency_completed, in_emergency_sequence, driver_has_slept
    global no_face_emergency_triggered, no_face_emergency_start_time

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    # Start emergency sequence when driver sleeps (any time, not just first time)
    if not in_emergency_sequence and current_driver_state == "DRIVER_SLEPT":
        in_emergency_sequence = True
        sleep_start_time_car = time.time()
        driver_has_slept = True
        print("Starting emergency sequence...")

    # Handle no face emergency
    if current_driver_state == "NO_FACE" and no_face_start_time:
        no_face_duration = time.time() - no_face_start_time

        if no_face_duration >= NO_FACE_EMERGENCY_THRESHOLD and not no_face_emergency_triggered:
            no_face_emergency_triggered = True
            no_face_emergency_start_time = time.time()
            print("NO FACE EMERGENCY: Stopping car with indicators!")

    # Reset no face emergency when face is detected again
    if current_driver_state != "NO_FACE" and no_face_emergency_triggered:
        no_face_emergency_triggered = False
        no_face_emergency_start_time = None
        print("NO FACE EMERGENCY: Face detected, resuming normal operation")

    if in_emergency_sequence:
        if current_driver_state == "DRIVER_SLEPT":
            # Driver is still asleep - continue emergency
            sleep_duration = time.time() - sleep_start_time_car

            if sleep_duration < SLEEP_EMERGENCY_THRESHOLD_CAR:
                # Countdown phase - maintain normal driving but STAY IN LEFT LANE if driver has slept before
                if car_speed < max_speed:
                    car_speed += acceleration

                # FIX: Always stay in left lane if driver has slept at least once
                if driver_has_slept:
                    target_x = left_lane
                else:
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
                        # Move to left lane with indicators blinking (only if not already in left lane)
                        if not driver_has_slept:  # Only move to left lane if it's the first sleep
                            lane_progress = maneuver_progress / 0.5
                            target_x = center_lane + (left_lane - center_lane) * lane_progress
                        else:
                            # Already in left lane from previous sleep, just stay there
                            target_x = left_lane

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
                        # Recovery complete - reset emergency state for next time
                        emergency_triggered_car = False
                        emergency_completed = True
                        in_emergency_sequence = False
                        recovery_start_time = None
                        sleep_start_time_car = None
                        emergency_maneuver_start_time = None
                        print("Recovery complete! Continuing in left lane.")
    elif no_face_emergency_triggered:
        # No face emergency - stop the car with blinking indicators
        car_speed = 0
        target_x = left_lane  # Stay in current lane or move to left lane
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

# No face detection variables
no_face_start_time = None

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


def draw_status_panel(frame, text, position, bg_color, text_color=(255, 255, 255)):
    """Draw a status panel with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Draw background rectangle
    x, y = position
    padding = 10
    cv2.rectangle(frame,
                  (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding),
                  bg_color, -1)

    # Draw border
    cv2.rectangle(frame,
                  (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding),
                  (255, 255, 255), 2)

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


# -----------------------------
# MAIN LOOP - IMPROVED LAYOUT
# -----------------------------
def main():
    global sleep_start_time, emergency_triggered, current_driver_state, car_running
    global sms_sent, sms_test_completed, no_face_start_time, no_face_sms_sent

    print("Starting Driver Monitoring System with NodeMCU AP Connection and SMS")
    print(f"NodeMCU AP: {NODEMCU_AP_IP}:{NODEMCU_AP_PORT}")
    print(f"SMS Emergency Threshold: {SMS_EMERGENCY_THRESHOLD} seconds")
    print(f"SMS No Face Threshold: {SMS_NO_FACE_THRESHOLD} seconds")
    print(f"No Face Emergency Stop Threshold: {NO_FACE_EMERGENCY_THRESHOLD} seconds")
    print("Make sure your computer is connected to the NodeMCU Access Point!")
    print("SMS will be sent when:")
    print("  - Driver is asleep for 10+ seconds")
    print("  - No face detected for 10+ seconds")
    print("Car will stop when:")
    print("  - No face detected for 10+ seconds")
    print("Press 'T' to manually test SMS functionality")

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

            # Create a copy for drawing to avoid modifying original
            display_frame = frame.copy()

            # Add gradient background for text areas
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 300), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

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

                # Reset no face detection when face is detected
                if no_face_start_time is not None:
                    no_face_duration = time.time() - no_face_start_time
                    if no_face_duration > 1:  # Only log if no face was significant
                        print(f"Face detected after {no_face_duration:.1f}s of no face")
                    no_face_start_time = None
                    no_face_sms_sent = False  # Reset SMS flag when face is detected again

                # Send simplified state to NodeMCU
                send_udp(combined_state)

                # Visual overlays with improved layout
                y_offset = 40
                line_height = 35

                # Eye metrics
                draw_status_panel(display_frame, f"L_EAR: {leftEAR:.2f}", (20, y_offset),
                                  (0, 100, 0), (0, 255, 0))
                draw_status_panel(display_frame, f"R_EAR: {rightEAR:.2f}", (20, y_offset + line_height),
                                  (0, 100, 100), (0, 255, 255))

                # Eye state with color coding
                eye_color = (0, 150, 255) if eye_state == "OPEN" else (0, 0, 255)
                draw_status_panel(display_frame, f"EYES: {eye_state}", (20, y_offset + line_height * 2),
                                  (50, 50, 50), eye_color)

                # Head direction
                head_color = (255, 0, 255)
                draw_status_panel(display_frame, f"HEAD: {head_direction}", (20, y_offset + line_height * 3),
                                  (50, 50, 50), head_color)

                # Head angles
                draw_status_panel(display_frame, f"X: {x_angle:.1f}", (20, y_offset + line_height * 4),
                                  (0, 0, 100), (255, 255, 255))
                draw_status_panel(display_frame, f"Y: {y_angle:.1f}", (120, y_offset + line_height * 4),
                                  (0, 0, 100), (255, 255, 255))
                draw_status_panel(display_frame, f"Z: {z_angle:.1f}", (220, y_offset + line_height * 4),
                                  (0, 0, 100), (255, 255, 255))

                # Main state with prominent display
                status_color = (0, 0, 200) if slept_detected else (0, 200, 0)
                status_bg = (0, 0, 100) if slept_detected else (0, 100, 0)
                draw_status_panel(display_frame, f"STATE: {combined_state}",
                                  (display_frame.shape[1] - 300, y_offset),
                                  status_bg, status_color)

                if slept_detected and sleep_start_time is not None:
                    sleep_duration = time.time() - sleep_start_time
                    draw_status_panel(display_frame, f"SLEEP TIMER: {sleep_duration:.1f}s",
                                      (display_frame.shape[1] - 300, y_offset + line_height),
                                      (0, 0, 100), (0, 0, 255))

                    # Show SMS status
                    if sleep_duration >= SMS_EMERGENCY_THRESHOLD:
                        sms_status = "SMS: SENT" if sms_sent else "SMS: PENDING"
                        sms_color = (0, 255, 0) if sms_sent else (0, 255, 255)
                        sms_bg = (0, 100, 0) if sms_sent else (0, 100, 100)
                        draw_status_panel(display_frame, sms_status,
                                          (display_frame.shape[1] - 300, y_offset + line_height * 2),
                                          sms_bg, sms_color)

                    if emergency_triggered:
                        draw_status_panel(display_frame, "EMERGENCY! SAFETY MANEUVER ACTIVATED",
                                          (display_frame.shape[1] // 2 - 200, y_offset + line_height * 5),
                                          (0, 0, 100), (0, 0, 255))
                    else:
                        remaining_time = max(0, SLEEP_EMERGENCY_THRESHOLD - sleep_duration)
                        draw_status_panel(display_frame, f"DRIVER ASLEEP! Emergency in {remaining_time:.1f}s",
                                          (display_frame.shape[1] // 2 - 200, y_offset + line_height * 5),
                                          (0, 50, 100), (0, 255, 255))

                # Draw head pose direction line
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
                cv2.line(display_frame, p1, p2, (255, 0, 0), 3)
                cv2.circle(display_frame, p1, 5, (0, 255, 0), -1)

                # Draw face landmarks
                mp_draw.draw_landmarks(
                    display_frame,
                    results.multi_face_landmarks[0],
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

            else:
                # No face detected
                combined_state = "NO_FACE"
                current_driver_state = "NO_FACE"
                send_udp("NO_FACE")

                # Start no face timer if not already started
                if no_face_start_time is None:
                    no_face_start_time = time.time()
                    print(f"No face detected at {time.strftime('%H:%M:%S')}")

                # Calculate no face duration
                no_face_duration = time.time() - no_face_start_time

                # Check if we need to send SMS for no face
                if no_face_duration >= SMS_NO_FACE_THRESHOLD and not no_face_sms_sent:
                    print(f"⏰ No Face SMS Trigger: Duration {no_face_duration:.1f}s >= {SMS_NO_FACE_THRESHOLD}s")
                    sms_message = f"🚨 ALERT: Driver not in position for {no_face_duration:.1f}s! No face detected. Time: {time.strftime('%H:%M:%S')}"
                    sms_result = send_sms(SMS_API_KEY, SMS_NUMBERS, sms_message)
                    no_face_sms_sent = sms_result  # Only mark as sent if successful
                    if sms_result:
                        print(f"✓ No Face SMS sent after {no_face_duration:.1f} seconds")
                    else:
                        print(f"✗ Failed to send no face SMS")

                # Reset sleep detection when no face is detected
                sleep_start_time = None
                if emergency_triggered:
                    emergency_triggered = False

                # Display no face status
                draw_status_panel(display_frame, "NO FACE DETECTED",
                                  (display_frame.shape[1] // 2 - 100, 50),
                                  (0, 0, 100), (0, 0, 255))

                # Show no face timer and SMS status
                draw_status_panel(display_frame, f"NO FACE TIMER: {no_face_duration:.1f}s",
                                  (display_frame.shape[1] // 2 - 100, 100),
                                  (0, 0, 100), (0, 0, 255))

                if no_face_duration >= SMS_NO_FACE_THRESHOLD:
                    no_face_sms_status = "NO FACE SMS: SENT" if no_face_sms_sent else "NO FACE SMS: PENDING"
                    no_face_sms_color = (0, 255, 0) if no_face_sms_sent else (0, 255, 255)
                    no_face_sms_bg = (0, 100, 0) if no_face_sms_sent else (0, 100, 100)
                    draw_status_panel(display_frame, no_face_sms_status,
                                      (display_frame.shape[1] // 2 - 100, 150),
                                      no_face_sms_bg, no_face_sms_color)

                # Show emergency stop warning
                if no_face_duration >= NO_FACE_EMERGENCY_THRESHOLD:
                    draw_status_panel(display_frame, "🚨 EMERGENCY STOP ACTIVATED!",
                                      (display_frame.shape[1] // 2 - 150, 200),
                                      (0, 0, 100), (0, 0, 255))
                else:
                    remaining_time = max(0, NO_FACE_EMERGENCY_THRESHOLD - no_face_duration)
                    draw_status_panel(display_frame, f"Emergency stop in {remaining_time:.1f}s",
                                      (display_frame.shape[1] // 2 - 100, 200),
                                      (0, 50, 100), (0, 255, 255))

            # Display FPS in top right
            draw_status_panel(display_frame, f'FPS: {int(fps)}',
                              (display_frame.shape[1] - 150, 10),
                              (50, 50, 50), (0, 255, 0))

            # System info
            info_text = f"NodeMCU AP: {NODEMCU_AP_IP}"
            draw_status_panel(display_frame, info_text,
                              (20, display_frame.shape[0] - 30),
                              (50, 50, 50), (255, 255, 255))

            # Manual SMS test instruction
            test_text = "Press 'T' to test SMS"
            draw_status_panel(display_frame, test_text,
                              (display_frame.shape[1] - 200, display_frame.shape[0] - 30),
                              (50, 50, 100), (255, 255, 255))

            cv2.imshow("Driver Monitoring System - Enhanced", display_frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                car_running = False
                break
            elif key == ord('t') or key == ord('T'):  # T key for manual SMS test
                print("\nManual SMS test triggered by user...")
                test_result = test_sms()
                if test_result:
                    print("Manual SMS test completed successfully!")
                else:
                    print("Manual SMS test failed!")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    car_running = False
    pygame.quit()


if __name__ == "__main__":
    main()
