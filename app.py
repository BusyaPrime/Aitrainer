import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, RTCConfiguration, webrtc_streamer
import av


def calculate_angle(a, b, c):
    """

    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = float(np.abs(radians * 180.0 / np.pi))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


class PoseDetector:
    def __init__(self, detection_con=0.5, track_con=0.5, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.results = None

    def find_pose(self, img, draw=True, color=(0, 0, 255)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        self.results = self.pose.process(img_rgb)
        img_rgb.flags.writeable = True

        if draw and self.results and self.results.pose_landmarks:
            spec = self.mp_draw.DrawingSpec(color=color, thickness=3, circle_radius=3)
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, spec, spec)
        return img

    def find_position(self, img):
        lm_list = []
        if not self.results or not self.results.pose_landmarks:
            return lm_list

        h, w = img.shape[:2]
        for idx, lm in enumerate(self.results.pose_landmarks.landmark):
            lm_list.append([idx, int(lm.x * w), int(lm.y * h)])
        return lm_list


def _overlay_box(img, x1, y1, x2, y2, color, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_ui(img, exercise, reps, state, angle, per, skeleton_green, warning=None):
    h, w = img.shape[:2]
    green, red = (0, 255, 0), (0, 0, 255)
    white, dark = (255, 255, 255), (15, 15, 15)
    accent = green if skeleton_green else red

    _overlay_box(img, 18, 18, 430, 145, dark, alpha=0.75)
    cv2.rectangle(img, (18, 18), (430, 145), accent, 2)

    cv2.putText(img, exercise, (34, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
    cv2.putText(img, "REPS", (34, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2, cv2.LINE_AA)
    cv2.putText(img, str(reps), (170, 130), cv2.FONT_HERSHEY_DUPLEX, 1.6, accent, 3, cv2.LINE_AA)

    st_txt = state if state else "--"
    cv2.putText(img, f"State: {st_txt}", (270, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2, cv2.LINE_AA)
    if angle is not None:
        cv2.putText(img, f"Angle: {angle:0.1f} deg", (270, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2, cv2.LINE_AA)

    bar_x, bar_top, bar_bottom, bar_w = w - 70, 90, h - 90, 30
    cv2.rectangle(img, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), white, 2)

    per = float(np.clip(per, 0.0, 100.0))
    fill_h = int((bar_bottom - bar_top) * (per / 100.0))
    fill_top = bar_bottom - fill_h
    cv2.rectangle(img, (bar_x + 2, fill_top), (bar_x + bar_w - 2, bar_bottom - 2), accent, -1)
    cv2.putText(img, f"{int(per)}%", (bar_x - 12, bar_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2, cv2.LINE_AA)

    if warning:
        _overlay_box(img, 18, h - 70, w - 18, h - 18, (0, 0, 0), alpha=0.55)
        cv2.putText(img, warning, (28, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2, cv2.LINE_AA)


def _lm_dict(lm_list):
    return {lm[0]: (lm[1], lm[2]) for lm in lm_list}


@dataclass
class ControlState:
    lock: threading.Lock
    exercise: str = "Bicep Curl Right"
    reset_flag: bool = False


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class TrainerVideoProcessor:
    def __init__(self, control: ControlState):
        self.control = control
        self.detector = PoseDetector(detection_con=0.5, track_con=0.5)

        self.lock = threading.Lock()
        self.reps = 0
        self.state: Optional[str] = None
        self.angle: Optional[float] = None
        self.progress = 0.0

    def _maybe_reset(self):
        with self.control.lock:
            if self.control.reset_flag:
                self.control.reset_flag = False
                with self.lock:
                    self.reps = 0
                    self.state = None
                    self.angle = None
                    self.progress = 0.0

    def _get_exercise(self) -> str:
        with self.control.lock:
            return self.control.exercise

    def get_stats(self):
        with self.lock:
            return {"reps": self.reps, "state": self.state, "angle": self.angle, "progress": self.progress}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._maybe_reset()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (960, 540))

        exercise = self._get_exercise()

        with self.lock:
            skeleton_green = bool(self.state == "UP")
        sk_color = (0, 255, 0) if skeleton_green else (0, 0, 255)

        img = self.detector.find_pose(img, draw=True, color=sk_color)
        lm_list = self.detector.find_position(img)

        warning = None
        angle = None
        per = 0.0

        if not lm_list:
            warning = "No person detected. Step into the frame."
        else:
            lms = _lm_dict(lm_list)

            if exercise.startswith("Bicep Curl"):
                if exercise.endswith("Left"):
                    s_id, e_id, w_id = 11, 13, 15
                else:
                    s_id, e_id, w_id = 12, 14, 16

                if s_id in lms and e_id in lms and w_id in lms:
                    angle = calculate_angle(lms[s_id], lms[e_id], lms[w_id])
                    per = float(np.interp(angle, [30.0, 160.0], [100.0, 0.0]))
                    per = float(np.clip(per, 0.0, 100.0))

                    with self.lock:
                        if angle > 160.0:
                            self.state = "DOWN"
                        if angle < 30.0 and self.state == "DOWN":
                            self.state = "UP"
                            self.reps += 1
                else:
                    warning = "Arm landmarks not found. Keep the full arm visible."

            elif exercise == "Squat":
                angles = []
                if 23 in lms and 25 in lms and 27 in lms:
                    angles.append(calculate_angle(lms[23], lms[25], lms[27]))
                if 24 in lms and 26 in lms and 28 in lms:
                    angles.append(calculate_angle(lms[24], lms[26], lms[28]))

                if angles:
                    angle = float(np.mean(angles))
                    per = float(np.interp(angle, [90.0, 170.0], [100.0, 0.0]))
                    per = float(np.clip(per, 0.0, 100.0))

                    with self.lock:
                        if angle > 170.0:
                            self.state = "UP"
                        if angle < 90.0 and self.state == "UP":
                            self.state = "DOWN"
                            self.reps += 1
                else:
                    warning = "Leg landmarks not found. Step back so legs are visible."

        with self.lock:
            self.angle = angle
            self.progress = per
            skeleton_green = bool(self.state == "UP")
            reps = self.reps
            state = self.state

        _draw_ui(img, exercise.replace("Right", "").replace("Left", "").strip(), reps, state, angle, per, skeleton_green, warning=warning)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="AI Personal Trainer (Cloud)", layout="wide")
    st.title("AI Personal Trainer")
    st.caption("Cloud-ready WebRTC version (phone camera supported).")

    if "control" not in st.session_state:
        st.session_state.control = ControlState(lock=threading.Lock())
    control: ControlState = st.session_state.control

    with st.sidebar:
        st.header("Controls")
        exercise = st.selectbox("Exercise", ["Bicep Curl Right", "Bicep Curl Left", "Squat"])
        if st.button("Reset Counter", use_container_width=True):
            with control.lock:
                control.reset_flag = True

    with control.lock:
        control.exercise = exercise

    left, right = st.columns([2.2, 1])
    with left:
        webrtc_ctx = webrtc_streamer(
            key="ai-personal-trainer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: TrainerVideoProcessor(control),
            async_processing=True,
        )

    with right:
        stats_box = st.empty()
        st.info("Press **START** above and allow camera permission in your browser.")

    if webrtc_ctx.video_processor:
        while webrtc_ctx.state.playing:
            stats = webrtc_ctx.video_processor.get_stats()
            with stats_box.container():
                st.metric("Reps", stats["reps"])
                st.metric("State", stats["state"] if stats["state"] else "--")
                st.metric("Angle (deg)", f'{stats["angle"]:.1f}' if stats["angle"] is not None else "--")
                st.metric("Progress", f'{int(stats["progress"])}%')
            time.sleep(0.2)


if __name__ == "__main__":
    main()
