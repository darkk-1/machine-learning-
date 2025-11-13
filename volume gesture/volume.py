import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Setup audio control using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()  # Get volume min/max
min_vol = vol_range[0]
max_vol = vol_range[1]

# Webcam input
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Thumb tip = 4, Index tip = 8
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[20][1], lm_list[20][2]

            # Draw circles on thumb tip and index tip
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Calculate distance
            length = np.hypot(x2 - x1, y2 - y1)

            # Map the distance to the volume range
            vol = np.interp(length, [30, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Draw volume bar
            vol_bar = np.interp(length, [30, 200], [400, 150])
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'Volume: {int(np.interp(length, [30, 200], [0, 100]))} %',
                        (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
