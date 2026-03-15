import cv2
import numpy as np
import pygame
import serial
import time
import random
import json
from dataclasses import dataclass
from pathlib import Path
from serial.tools import list_ports
from config import CONFIG


@dataclass
class Balloon:
    x: float
    y: float
    radius: int
    speed: float
    kind: str
    alive: bool = True

    def update(self, dt):
        self.y -= self.speed * dt
        if self.y < -self.radius:
            self.alive = False

    def score(self):
        if self.kind == "normal":
            return CONFIG["score_normal"]
        if self.kind == "bonus":
            return CONFIG["score_bonus"]
        if self.kind == "bomb":
            return CONFIG["score_bomb"]
        return 0


@dataclass
class PopParticle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: tuple
    life: float

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += CONFIG["pop_particle_gravity"] * dt
        self.life -= dt


@dataclass
class PopBurst:
    x: float
    y: float
    radius: float
    color: tuple
    life: float
    max_life: float
    particles: list

    def update(self, dt):
        self.life -= dt

        for particle in self.particles:
            particle.update(dt)

        self.particles = [particle for particle in self.particles if particle.life > 0]

    def alive(self):
        return self.life > 0 or bool(self.particles)


class LaserTracker:
    def __init__(self):
        self.cap = self.open_camera()
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG.get("camera_buffer_size", 1))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.get("camera_w", 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.get("camera_h", 720))
        self.apply_camera_settings()
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CONFIG.get(
            "camera_w",
            1280,
        )
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CONFIG.get(
            "camera_h",
            720,
        )
        self.cam_points = self.scale_cam_points()
        self.last_camera_point = None
        self.feedback_text = ""
        self.feedback_color = (255, 255, 255)
        self.feedback_until = 0
        self.feedback_camera_point = None
        self.screen_offset_x = CONFIG.get("laser_screen_offset_x", 0)
        self.screen_offset_y = CONFIG.get("laser_screen_offset_y", 0)
        self.last_screen_point = None
        self.last_detected_at = 0.0
        self.smoothed_screen_point = None
        self.selected_cam_point_index = 0
        self.calibration_mode = False
        self.last_frame = None
        self.auto_quad_points = None
        self.bg_response_model = None

        screen_points = np.array(
            [
                [0, 0],
                [CONFIG["screen_w"], 0],
                [CONFIG["screen_w"], CONFIG["screen_h"]],
                [0, CONFIG["screen_h"]],
            ],
            dtype=np.float32,
        )
        self.screen_points = screen_points

        self.update_h_matrix()

    def apply_camera_settings(self):
        camera_settings = [
            (cv2.CAP_PROP_AUTO_EXPOSURE, CONFIG.get("camera_auto_exposure")),
            (cv2.CAP_PROP_EXPOSURE, CONFIG.get("camera_exposure")),
            (cv2.CAP_PROP_GAIN, CONFIG.get("camera_gain")),
            (cv2.CAP_PROP_BRIGHTNESS, CONFIG.get("camera_brightness")),
        ]

        for prop_id, value in camera_settings:
            if value is None:
                continue
            try:
                self.cap.set(prop_id, value)
            except Exception:
                pass

        print(
            "Camera settings:",
            {
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "gain": self.cap.get(cv2.CAP_PROP_GAIN),
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            },
        )

    def update_h_matrix(self):
        self.h_matrix = cv2.getPerspectiveTransform(self.cam_points, self.screen_points)
        self.update_detection_roi()

    def update_detection_roi(self):
        points = self.cam_points.astype(np.int32)
        x, y, w, h = cv2.boundingRect(points)
        self.roi_x = x
        self.roi_y = y
        self.roi_w = max(1, w)
        self.roi_h = max(1, h)

        local_points = self.cam_points.copy()
        local_points[:, 0] -= self.roi_x
        local_points[:, 1] -= self.roi_y
        self.roi_points = local_points
        self.roi_mask = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(
            self.roi_mask,
            self.roi_points.astype(np.int32),
            255,
        )
        self.bg_response_model = None

    def order_points(self, points):
        points = np.array(points, dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered

    def backend_id(self, name):
        if name == "dshow":
            return cv2.CAP_DSHOW
        if name == "msmf":
            return cv2.CAP_MSMF
        if name == "any":
            return cv2.CAP_ANY
        return cv2.CAP_ANY

    def open_camera(self):
        backend_name = str(CONFIG.get("camera_backend", "any")).lower()
        if backend_name == "any":
            backends = ["any", "dshow", "msmf"]
        elif backend_name == "dshow":
            backends = ["dshow", "any", "msmf"]
        elif backend_name == "msmf":
            backends = ["msmf", "any", "dshow"]
        else:
            backends = ["any", "dshow", "msmf"]

        indices = [CONFIG.get("camera_index", 0)]

        for index in CONFIG.get("camera_probe_indices", []):
            if index not in indices:
                indices.append(index)

        for backend in backends:
            backend_id = self.backend_id(backend)

            for index in indices:
                cap = cv2.VideoCapture(index, backend_id)
                if cap.isOpened():
                    ok = False

                    for _ in range(15):
                        ok, _ = cap.read()
                        if ok:
                            break
                        time.sleep(0.03)

                    if ok:
                        print(f"Camera opened: index={index}, backend={backend}")
                        return cap
                cap.release()

        raise RuntimeError(
            f"Camera open failed. Tried indices={indices} backends={backends}"
        )

    def scale_cam_points(self):
        cam_points = np.array(CONFIG["cam_points"], dtype=np.float32)
        ref_w = CONFIG.get("cam_points_ref_w", self.frame_w)
        ref_h = CONFIG.get("cam_points_ref_h", self.frame_h)

        if ref_w <= 0 or ref_h <= 0:
            return cam_points

        scaled = cam_points.copy()
        scaled[:, 0] *= self.frame_w / ref_w
        scaled[:, 1] *= self.frame_h / ref_h
        return scaled

    def draw_calibration_overlay(self, image):
        if not CONFIG.get("show_calibration_overlay", False):
            return image

        points = self.cam_points.astype(np.int32)
        overlay = image.copy()

        cv2.polylines(overlay, [points], True, (255, 200, 0), 2)

        if self.auto_quad_points is not None:
            auto_points = self.auto_quad_points.astype(np.int32)
            cv2.polylines(overlay, [auto_points], True, (255, 0, 180), 2)

        for index, (x, y) in enumerate(points):
            point_color = (0, 255, 120) if index == self.selected_cam_point_index else (0, 200, 255)
            point_radius = 8 if index == self.selected_cam_point_index else 6
            cv2.circle(overlay, (x, y), point_radius, point_color, -1)
            cv2.putText(
                overlay,
                f"P{index + 1} ({x}, {y})",
                (x + 8, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                point_color,
                2,
            )

        cv2.putText(
            overlay,
            "C: calib  Tab: next  Arrow: move  A: auto  Ctrl+S: save",
            (20, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 200, 0),
            2,
        )

        cv2.putText(
            overlay,
            f"frame: {self.frame_w}x{self.frame_h}",
            (20, image.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 0),
            2,
        )

        cv2.putText(
            overlay,
            f"calibration: {'ON' if self.calibration_mode else 'OFF'}",
            (20, image.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 120) if self.calibration_mode else (180, 180, 180),
            2,
        )

        return overlay

    def toggle_calibration_mode(self):
        self.calibration_mode = not self.calibration_mode
        print("Calibration mode:", "ON" if self.calibration_mode else "OFF")

    def select_cam_point(self, index):
        if 0 <= index < len(self.cam_points):
            self.selected_cam_point_index = index
            print(f"Selected cam point: P{index + 1}")

    def select_next_cam_point(self):
        self.selected_cam_point_index = (
            self.selected_cam_point_index + 1
        ) % len(self.cam_points)
        print(f"Selected cam point: P{self.selected_cam_point_index + 1}")

    def move_selected_cam_point(self, dx, dy):
        point = self.cam_points[self.selected_cam_point_index]
        point[0] = np.clip(point[0] + dx, 0, self.frame_w - 1)
        point[1] = np.clip(point[1] + dy, 0, self.frame_h - 1)
        self.update_h_matrix()
        print(
            f"P{self.selected_cam_point_index + 1} = "
            f"({int(point[0])}, {int(point[1])})"
        )

    def cam_points_for_config(self):
        ref_w = CONFIG.get("cam_points_ref_w", self.frame_w)
        ref_h = CONFIG.get("cam_points_ref_h", self.frame_h)
        export_points = self.cam_points.copy()

        if self.frame_w > 0 and self.frame_h > 0:
            export_points[:, 0] *= ref_w / self.frame_w
            export_points[:, 1] *= ref_h / self.frame_h

        return [[int(round(x)), int(round(y))] for x, y in export_points]

    def save_calibration_to_config(self, config_path="config.py"):
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_points_block = "    \"cam_points\": [\n"
        for x, y in self.cam_points_for_config():
            new_points_block += f"        [{x}, {y}],\n"
        new_points_block += "    ],"

        import re

        content, count = re.subn(
            r'    "cam_points": \[\n(?:        \[[^\n]+\],\n)+    \],',
            new_points_block,
            content,
        )

        if count != 1:
            raise RuntimeError("Failed to update cam_points in config.py")

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("Saved tracker settings to config.py")

    def detect_auto_quad(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(
            blurred,
            CONFIG.get("auto_calibration_canny_low", 40),
            CONFIG.get("auto_calibration_canny_high", 120),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best_quad = None
        best_area = 0
        min_area = (
            self.frame_w
            * self.frame_h
            * CONFIG.get("auto_calibration_min_area_ratio", 0.12)
        )
        epsilon_ratio = CONFIG.get("auto_calibration_epsilon_ratio", 0.03)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, perimeter * epsilon_ratio, True)

            if len(approx) == 4 and area > best_area:
                best_area = area
                best_quad = self.order_points(approx.reshape(4, 2))

        if best_quad is None and contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area:
                rect = cv2.minAreaRect(largest)
                best_quad = self.order_points(cv2.boxPoints(rect))

        self.auto_quad_points = best_quad
        return best_quad

    def auto_calibrate(self):
        if self.last_frame is None:
            print("Auto calibration skipped: no camera frame yet")
            return False

        quad = self.detect_auto_quad(self.last_frame)
        if quad is None:
            print("Auto calibration failed: no screen edge quad found")
            return False

        self.cam_points = quad.astype(np.float32)
        self.update_h_matrix()
        print("Auto calibration applied:", self.cam_points_for_config())
        return True

    def build_laser_mask(self, frame):
        roi = frame[
            self.roi_y : self.roi_y + self.roi_h,
            self.roi_x : self.roi_x + self.roi_w,
        ]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bgr_int = roi.astype(np.int16)
        blue = bgr_int[:, :, 0]
        green = bgr_int[:, :, 1]
        red = bgr_int[:, :, 2]
        red_margin = CONFIG.get("laser_red_margin", 30)
        red_min = CONFIG.get("laser_r_min", 90)

        red_excess = np.clip(red - np.maximum(green, blue), 0, 255).astype(np.uint8)
        dominant_score = np.clip(red * 2 - green - blue, 0, 255).astype(np.uint8)
        bright_score = gray.astype(np.uint8)

        bias_kernel = int(CONFIG.get("laser_column_bias_kernel", 81))
        if bias_kernel % 2 == 0:
            bias_kernel += 1
        bias_gain = float(CONFIG.get("laser_column_bias_gain", 0.8))
        column_profile = bright_score.mean(axis=0, keepdims=True).astype(np.float32)
        smooth_profile = cv2.GaussianBlur(column_profile, (bias_kernel, 1), 0)
        column_bias = smooth_profile - smooth_profile.min()
        bias_image = np.repeat(
            np.clip(column_bias * bias_gain, 0, 255).astype(np.uint8),
            bright_score.shape[0],
            axis=0,
        )
        bright_score = cv2.subtract(bright_score, bias_image)

        kernel_size = int(CONFIG.get("laser_local_bg_kernel", 9))
        if kernel_size % 2 == 0:
            kernel_size += 1

        local_bg = cv2.GaussianBlur(bright_score, (kernel_size, kernel_size), 0)
        local_peak = cv2.subtract(bright_score, local_bg)
        raw_gain = float(CONFIG.get("laser_raw_response_gain", 0.55))
        brightness_gain = float(CONFIG.get("laser_brightness_gain", 1.0))
        red_gain = float(CONFIG.get("laser_red_response_gain", 1.0))
        boosted_raw = np.clip(bright_score.astype(np.float32) * brightness_gain, 0, 255).astype(
            np.uint8
        )
        red_bg = cv2.GaussianBlur(dominant_score, (kernel_size, kernel_size), 0)
        red_local_peak = cv2.subtract(dominant_score, red_bg)
        boosted_red = np.clip(dominant_score.astype(np.float32) * red_gain, 0, 255).astype(
            np.uint8
        )

        dominance_mask = (
            (red >= red_min)
            & (red - green >= red_margin)
            & (red - blue >= red_margin)
        ).astype(np.uint8) * 255

        bright_mask = (bright_score >= CONFIG.get("laser_v_min", 90)).astype(np.uint8) * 255

        valid_mask = cv2.bitwise_and(
            cv2.bitwise_or(bright_mask, dominance_mask),
            self.roi_mask,
        )

        left_ignore = int(CONFIG.get("laser_ignore_left_px", 0))
        right_ignore = int(CONFIG.get("laser_ignore_right_px", 0))
        top_ignore = int(CONFIG.get("laser_ignore_top_px", 0))
        bottom_ignore = int(CONFIG.get("laser_ignore_bottom_px", 0))

        if left_ignore > 0:
            valid_mask[:, : min(left_ignore, self.roi_w)] = 0
        if right_ignore > 0:
            valid_mask[:, max(0, self.roi_w - right_ignore) :] = 0
        if top_ignore > 0:
            valid_mask[: min(top_ignore, self.roi_h), :] = 0
        if bottom_ignore > 0:
            valid_mask[max(0, self.roi_h - bottom_ignore) :, :] = 0

        if self.bg_response_model is None or self.bg_response_model.shape != bright_score.shape:
            self.bg_response_model = bright_score.astype(np.float32)

        bg_alpha = float(CONFIG.get("laser_bg_alpha", 0.04))
        temporal_gain = float(CONFIG.get("laser_temporal_gain", 0.65))
        temporal_delta = cv2.subtract(
            bright_score,
            self.bg_response_model.astype(np.uint8),
        )
        temporal_boost = np.clip(
            temporal_delta.astype(np.float32) * temporal_gain,
            0,
            255,
        ).astype(np.uint8)

        response = cv2.max(local_peak, boosted_raw)
        response = cv2.max(response, temporal_boost)
        response = cv2.max(response, red_local_peak)
        response = cv2.max(response, boosted_red)
        response = cv2.max(
            response,
            np.clip(dominant_score.astype(np.float32) * raw_gain, 0, 255).astype(np.uint8),
        )
        response = cv2.GaussianBlur(response, (3, 3), 0)
        response = cv2.bitwise_and(response, valid_mask)

        masked_bright = bright_score.astype(np.float32)
        masked_bright[valid_mask == 0] = 0
        self.bg_response_model = (
            self.bg_response_model * (1.0 - bg_alpha)
            + masked_bright * bg_alpha
        )

        mask = valid_mask

        full_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        full_mask[
            self.roi_y : self.roi_y + self.roi_h,
            self.roi_x : self.roi_x + self.roi_w,
        ] = mask

        return full_mask, response

    def detect_best_laser_point(self, mask, response):
        local_mask = mask[
            self.roi_y : self.roi_y + self.roi_h,
            self.roi_x : self.roi_x + self.roi_w,
        ]
        response_min = CONFIG.get("laser_response_min", 10)
        _, peak_score, _, peak_loc = cv2.minMaxLoc(response, mask=local_mask)
        candidate_score = peak_score
        candidate_loc = peak_loc

        if self.last_camera_point is not None:
            search_radius = int(CONFIG.get("laser_tracking_search_radius", 90))
            min_ratio = float(CONFIG.get("laser_tracking_min_score_ratio", 0.6))
            relaxed_min = float(CONFIG.get("laser_tracking_relaxed_min", 3))
            prev_x = int(self.last_camera_point[0] - self.roi_x)
            prev_y = int(self.last_camera_point[1] - self.roi_y)

            search_mask = np.zeros_like(local_mask)
            cv2.circle(search_mask, (prev_x, prev_y), search_radius, 255, -1)
            local_search_mask = cv2.bitwise_and(local_mask, search_mask)
            _, local_score, _, local_loc = cv2.minMaxLoc(response, mask=local_search_mask)

            if local_score >= response_min * min_ratio:
                candidate_score = local_score
                candidate_loc = local_loc
            elif candidate_score < response_min and local_score >= relaxed_min:
                candidate_score = local_score
                candidate_loc = local_loc

        if candidate_score < response_min:
            return None, candidate_score

        px, py = candidate_loc
        radius = int(CONFIG.get("laser_peak_window_radius", 4))
        x0 = max(0, px - radius)
        y0 = max(0, py - radius)
        x1 = min(self.roi_w, px + radius + 1)
        y1 = min(self.roi_h, py + radius + 1)

        patch_response = response[y0:y1, x0:x1].astype(np.float32)
        patch_mask = local_mask[y0:y1, x0:x1] > 0
        relative_threshold = candidate_score * float(
            CONFIG.get("laser_peak_relative_threshold", 0.45)
        )
        strong_patch = patch_response >= relative_threshold
        weights = np.where(patch_mask & strong_patch, patch_response, 0.0)
        total = float(weights.sum())

        if total > 0:
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            cx = float((weights * grid_x).sum() / total)
            cy = float((weights * grid_y).sum() / total)
        else:
            cx = float(px)
            cy = float(py)

        return (self.roi_x + cx, self.roi_y + cy), candidate_score

    def apply_screen_offset(self, sx, sy):
        sx += self.screen_offset_x
        sy += self.screen_offset_y
        sx = float(np.clip(sx, 0, CONFIG["screen_w"] - 1))
        sy = float(np.clip(sy, 0, CONFIG["screen_h"] - 1))
        return sx, sy

    def adjust_screen_offset(self, dx, dy):
        self.screen_offset_x += dx
        self.screen_offset_y += dy
        print(
            "Laser screen offset:",
            f"x={self.screen_offset_x}",
            f"y={self.screen_offset_y}",
        )

    def debug_lines(self):
        return [
            f"Aim offset: x={self.screen_offset_x} y={self.screen_offset_y}",
            "Ctrl+Arrow: adjust aim",
            f"Cam point: P{self.selected_cam_point_index + 1}",
            f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}",
            "C calib  Tab next  Arrow move  A auto  Ctrl+S save",
        ]

    def set_shot_feedback(self, hit):
        self.feedback_text = "HIT" if hit else "NO HIT"
        self.feedback_color = (
            CONFIG["shot_hit_color"] if hit else CONFIG["shot_miss_color"]
        )
        self.feedback_until = time.time() + CONFIG["camera_feedback_duration_sec"]
        self.feedback_camera_point = self.last_camera_point

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        return frame, time.time()

    def process_frame(self, frame, frame_time=None, use_smoothing=True):
        if frame_time is None:
            frame_time = time.time()

        self.last_frame = frame.copy()

        camera_view = self.draw_calibration_overlay(frame.copy())
        tracking_view = camera_view.copy()
        thresh, response = self.build_laser_mask(frame)
        best, best_score = self.detect_best_laser_point(thresh, response)

        screen_point = None
        self.last_camera_point = None

        if best is not None:
            cx, cy = best
            self.last_camera_point = (int(cx), int(cy))
            src = np.array([[[cx, cy]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.h_matrix)
            sx, sy = dst[0][0]
            sx, sy = self.apply_screen_offset(sx, sy)
            if use_smoothing:
                smoothing = float(CONFIG.get("laser_screen_smoothing", 0.35))
                if self.smoothed_screen_point is None:
                    self.smoothed_screen_point = (sx, sy)
                else:
                    prev_x, prev_y = self.smoothed_screen_point
                    self.smoothed_screen_point = (
                        prev_x * (1.0 - smoothing) + sx * smoothing,
                        prev_y * (1.0 - smoothing) + sy * smoothing,
                    )
                screen_point = self.smoothed_screen_point
            else:
                screen_point = (sx, sy)
                self.smoothed_screen_point = screen_point
            self.last_screen_point = screen_point
            self.last_detected_at = frame_time

            cv2.circle(camera_view, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            cv2.putText(
                camera_view,
                f"LASER ({int(cx)}, {int(cy)})",
                (int(cx) + 10, max(25, int(cy) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            cv2.circle(tracking_view, (int(cx), int(cy)), 12, (0, 255, 255), 2)
            cv2.putText(
                tracking_view,
                f"laser: ({int(cx)}, {int(cy)})",
                (int(cx) + 12, max(25, int(cy) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                tracking_view,
                f"screen: ({int(sx)}, {int(sy)}) score:{int(best_score)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            if (
                self.last_screen_point is not None
                and frame_time - self.last_detected_at <= CONFIG.get("laser_hold_sec", 0.08)
            ):
                screen_point = self.last_screen_point
            else:
                self.smoothed_screen_point = None

            cv2.putText(
                tracking_view,
                "laser: not found",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            tracking_view,
            f"offset: ({self.screen_offset_x}, {self.screen_offset_y})",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 200, 255),
            2,
        )

        if time.time() < self.feedback_until:
            cv2.putText(
                camera_view,
                self.feedback_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                self.feedback_color,
                2,
            )

            cv2.putText(
                tracking_view,
                self.feedback_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                self.feedback_color,
                2,
            )

            if self.feedback_camera_point is not None:
                cv2.circle(
                    camera_view,
                    self.feedback_camera_point,
                    18,
                    self.feedback_color,
                    3,
                )
                cv2.circle(
                    tracking_view,
                    self.feedback_camera_point,
                    18,
                    self.feedback_color,
                    3,
                )

        if CONFIG["show_camera_window"]:
            cv2.imshow("camera", camera_view)
            cv2.waitKey(1)

        if CONFIG["show_threshold_window"]:
            cv2.imshow("threshold", thresh)
            cv2.waitKey(1)

        if CONFIG.get("show_tracking_window", False):
            tracking_scale = float(CONFIG.get("tracking_window_scale", 1.0))
            tracking_w = max(320, int(self.frame_w * tracking_scale))
            tracking_h = max(180, int(self.frame_h * tracking_scale))
            cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("tracking", tracking_w, tracking_h)
            cv2.imshow("tracking", tracking_view)
            cv2.waitKey(1)

        return screen_point

    def read(self):
        frame, frame_time = self.read_frame()
        if frame is None:
            return None
        return self.process_frame(frame, frame_time=frame_time, use_smoothing=True)

    def read_for_shot(self):
        shot_grabs = max(0, int(CONFIG.get("shot_frame_grabs", 2)))

        for _ in range(shot_grabs):
            try:
                if not self.cap.grab():
                    break
            except Exception:
                break

        frame, frame_time = self.read_frame()
        if frame is None:
            return self.last_screen_point

        return self.process_frame(frame, frame_time=frame_time, use_smoothing=False)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


class ShotReceiver:
    def __init__(self):
        self.use_serial = CONFIG["use_serial"]
        self.ser = None
        self.last_shot = 0
        self.port_name = None
        self.status_text = "Serial: disabled"
        self.last_serial_line = ""

        if self.use_serial:
            self.open_serial()

    def serial_candidates(self):
        candidates = []
        configured_port = CONFIG.get("serial_port")

        if configured_port:
            candidates.append(configured_port)

        for port in list_ports.comports():
            if port.device not in candidates:
                candidates.append(port.device)

        return candidates

    def open_serial(self):
        errors = []

        for port_name in self.serial_candidates():
            try:
                self.ser = serial.Serial(
                    port_name,
                    CONFIG["baudrate"],
                    timeout=0.01,
                )
                self.port_name = port_name
                time.sleep(2)
                self.ser.reset_input_buffer()
                self.status_text = f"Serial: {port_name}"
                return
            except Exception as e:
                errors.append(f"{port_name}: {e}")

        self.use_serial = False

        if errors:
            self.status_text = f"Serial OFF: {errors[0]}"
        else:
            self.status_text = "Serial OFF: no ports found"

    def poll_serial(self):
        if not (self.use_serial and self.ser):
            return False

        shot_detected = False

        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                self.last_serial_line = line

                if line == "SHOT":
                    shot_detected = True
        except Exception as e:
            self.status_text = f"Serial OFF: {e}"
            self.use_serial = False

            if self.ser is not None:
                self.ser.close()
                self.ser = None

        return shot_detected

    def poll(self):
        now = time.time()
        serial_shot = self.poll_serial()

        if now - self.last_shot < CONFIG["shot_cooldown_sec"]:
            return False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.last_shot = now
            return True

        if serial_shot:
            self.last_shot = now
            return True

        return False

    def debug_lines(self):
        lines = [self.status_text]

        if self.last_serial_line:
            lines.append(f"RX: {self.last_serial_line}")

        return lines

    def release(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None


class SoundManager:
    def __init__(self):
        self.enabled = CONFIG.get("enable_sound", True)
        self.sounds = {}
        self.bgm_channel = None
        self.bgm_path = Path(__file__).with_name(CONFIG.get("bgm_file", "Energy.mp3"))

        if not self.enabled:
            return

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2)

            pygame.mixer.set_num_channels(12)
            pygame.mixer.set_reserved(1)
            self.bgm_channel = pygame.mixer.Channel(0)

            self.sounds = {
                "shot": self.make_shot_sound(),
                "pop": self.make_pop_sound(),
                "miss": self.make_miss_sound(),
                "game_over": self.make_game_over_sound(),
            }
            if not self.bgm_path.exists():
                self.sounds["bgm"] = self.make_bgm_loop()
            self.start_bgm()
        except pygame.error:
            self.enabled = False

    def audio_context(self):
        mixer_config = pygame.mixer.get_init()
        sample_rate = mixer_config[0] if mixer_config else 44100
        channels = mixer_config[2] if mixer_config else 2
        return sample_rate, channels

    def build_sound(self, waveform, volume):
        _, channels = self.audio_context()
        master_volume = CONFIG.get("sound_volume", 0.35) * volume
        audio = np.clip(waveform * 32767 * master_volume, -32768, 32767).astype(np.int16)

        if channels > 1:
            audio = np.repeat(audio[:, np.newaxis], channels, axis=1)

        return pygame.sndarray.make_sound(audio)

    def build_envelope(self, sample_count, sample_rate, attack_sec, decay_rate):
        attack_count = max(1, int(sample_rate * attack_sec))
        envelope = np.exp(-np.linspace(0, decay_rate, sample_count))
        envelope[:attack_count] = np.linspace(0.0, 1.0, attack_count)
        return envelope

    def make_shot_sound(self):
        sample_rate, _ = self.audio_context()
        duration_sec = CONFIG.get("shot_sound_duration_sec", 0.09)
        sample_count = max(1, int(sample_rate * duration_sec))
        freqs = np.linspace(2200, 540, sample_count)
        phase = np.cumsum(2 * np.pi * freqs / sample_rate)
        tone = 0.82 * np.sin(phase) + 0.18 * np.sin(phase * 1.8)
        envelope = self.build_envelope(sample_count, sample_rate, 0.002, 8.5)
        waveform = tone * envelope
        return self.build_sound(waveform, CONFIG.get("shot_sound_volume", 0.92))

    def make_pop_sound(self):
        sample_rate, _ = self.audio_context()
        duration_sec = CONFIG.get("pop_sound_duration_sec", 0.22)
        sample_count = max(1, int(sample_rate * duration_sec))
        freqs = np.linspace(520, 140, sample_count)
        phase = np.cumsum(2 * np.pi * freqs / sample_rate)
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 1.0, sample_count)
        noise = np.convolve(noise, np.ones(9) / 9, mode="same")
        tone = np.sin(phase) + 0.32 * np.sin(phase * 1.7)
        body_envelope = self.build_envelope(sample_count, sample_rate, 0.001, 5.5)
        crack_envelope = self.build_envelope(sample_count, sample_rate, 0.0005, 15.0)
        waveform = 0.72 * tone * body_envelope + 0.48 * noise * crack_envelope
        return self.build_sound(waveform, CONFIG.get("pop_sound_volume", 1.0))

    def make_miss_sound(self):
        sample_rate, _ = self.audio_context()
        duration_sec = 0.11
        sample_count = max(1, int(sample_rate * duration_sec))
        freqs = np.linspace(420, 240, sample_count)
        phase = np.cumsum(2 * np.pi * freqs / sample_rate)
        tone = np.sin(phase)
        envelope = self.build_envelope(sample_count, sample_rate, 0.003, 6.2)
        waveform = tone * envelope
        return self.build_sound(waveform, 0.22)

    def make_game_over_sound(self):
        sample_rate, _ = self.audio_context()
        parts = []
        for start_freq, end_freq, duration_sec in (
            (240, 170, 0.18),
            (180, 120, 0.24),
        ):
            sample_count = max(1, int(sample_rate * duration_sec))
            freqs = np.linspace(start_freq, end_freq, sample_count)
            phase = np.cumsum(2 * np.pi * freqs / sample_rate)
            tone = np.sin(phase) + 0.2 * np.sin(phase * 0.5)
            envelope = self.build_envelope(sample_count, sample_rate, 0.004, 4.5)
            parts.append(tone * envelope)
        waveform = np.concatenate(parts)
        return self.build_sound(waveform, 0.4)

    def make_bgm_loop(self):
        sample_rate, _ = self.audio_context()
        step_sec = CONFIG.get("bgm_step_sec", 0.32)
        note_sec = CONFIG.get("bgm_note_sec", 0.26)
        melody = [
            659.25,
            783.99,
            880.00,
            783.99,
            659.25,
            783.99,
            987.77,
            783.99,
            587.33,
            659.25,
            783.99,
            659.25,
            523.25,
            659.25,
            783.99,
            987.77,
        ]
        bassline = [
            164.81,
            164.81,
            146.83,
            146.83,
            174.61,
            174.61,
            146.83,
            146.83,
        ]
        duration_sec = step_sec * len(melody)
        sample_count = max(1, int(sample_rate * duration_sec))
        timeline = np.linspace(0, duration_sec, sample_count, False)
        waveform = np.zeros(sample_count, dtype=np.float32)
        rng = np.random.default_rng(11)

        # Steady low synth bed.
        waveform += 0.12 * np.sin(2 * np.pi * 110.0 * timeline)
        waveform += 0.07 * np.sin(2 * np.pi * 220.0 * timeline + 0.25)

        # Driving bass pulse.
        bass_step = max(1, len(melody) // len(bassline))
        for index, frequency in enumerate(bassline):
            start = int(index * bass_step * step_sec * sample_rate)
            end = min(sample_count, start + int((step_sec * bass_step) * sample_rate))
            if end <= start:
                continue
            note_t = np.linspace(0, (end - start) / sample_rate, end - start, False)
            phase = 2 * np.pi * frequency * note_t
            pulse = np.sign(np.sin(phase))
            tone = 0.16 * pulse + 0.08 * np.sin(phase)
            envelope = np.exp(-np.linspace(0, 4.6, end - start))
            waveform[start:end] += tone * envelope

        for index, frequency in enumerate(melody):
            start = int(index * step_sec * sample_rate)
            end = min(sample_count, start + int(note_sec * sample_rate))
            if end <= start:
                continue
            note_t = np.linspace(0, (end - start) / sample_rate, end - start, False)
            phase = 2 * np.pi * frequency * note_t
            pulse = np.sign(np.sin(phase))
            note = 0.16 * pulse + 0.20 * np.sin(phase) + 0.07 * np.sin(phase * 2.0)
            note_envelope = np.exp(-np.linspace(0, 3.8, end - start))
            attack_count = max(1, int(sample_rate * 0.01))
            note_envelope[:attack_count] = np.linspace(0.0, 1.0, attack_count)
            waveform[start:end] += note * note_envelope

            # Off-beat sparkle to keep the loop moving.
            arp_start = start + int(step_sec * 0.5 * sample_rate)
            arp_end = min(sample_count, arp_start + int(note_sec * 0.55 * sample_rate))
            if arp_end > arp_start:
                arp_t = np.linspace(0, (arp_end - arp_start) / sample_rate, arp_end - arp_start, False)
                arp_phase = 2 * np.pi * (frequency * 2.0) * arp_t
                arp = 0.10 * np.sin(arp_phase) + 0.04 * np.sin(arp_phase * 1.5)
                arp_env = np.exp(-np.linspace(0, 5.0, arp_end - arp_start))
                waveform[arp_start:arp_end] += arp * arp_env

        # Lightweight kick and hi-hat for arcade momentum.
        for beat_index in range(len(melody)):
            beat_start = int(beat_index * step_sec * sample_rate)

            if beat_index % 2 == 0:
                kick_end = min(sample_count, beat_start + int(0.10 * sample_rate))
                kick_t = np.linspace(0, (kick_end - beat_start) / sample_rate, kick_end - beat_start, False)
                kick_freq = np.linspace(120.0, 42.0, kick_end - beat_start)
                kick_phase = np.cumsum(2 * np.pi * kick_freq / sample_rate)
                kick = 0.20 * np.sin(kick_phase)
                kick_env = np.exp(-np.linspace(0, 8.0, kick_end - beat_start))
                waveform[beat_start:kick_end] += kick * kick_env

            hat_start = beat_start + int(step_sec * 0.5 * sample_rate)
            hat_end = min(sample_count, hat_start + int(0.035 * sample_rate))
            if hat_end > hat_start:
                hat = rng.normal(0.0, 1.0, hat_end - hat_start)
                hat = np.convolve(hat, np.array([1.0, -0.85]), mode="same")
                hat_env = np.exp(-np.linspace(0, 10.0, hat_end - hat_start))
                waveform[hat_start:hat_end] += 0.055 * hat * hat_env

        fade_count = max(1, int(sample_rate * 0.05))
        waveform[:fade_count] *= np.linspace(0.0, 1.0, fade_count)
        waveform[-fade_count:] *= np.linspace(1.0, 0.0, fade_count)
        waveform = np.tanh(waveform * 1.4)
        return self.build_sound(waveform, CONFIG.get("bgm_volume", 0.24))

    def play(self, name):
        if not self.enabled:
            return

        sound = self.sounds.get(name)
        if sound is not None:
            channel = pygame.mixer.find_channel()
            if channel is not None:
                channel.play(sound)
            else:
                sound.play()

    def start_bgm(self):
        if not self.enabled or not CONFIG.get("enable_bgm", True):
            return

        if self.bgm_path.exists():
            try:
                pygame.mixer.music.load(str(self.bgm_path))
                pygame.mixer.music.set_volume(CONFIG.get("bgm_volume", 0.28))
                pygame.mixer.music.play(-1)
                return
            except pygame.error:
                pass

        if self.bgm_channel is None:
            return

        bgm = self.sounds.get("bgm")
        if bgm is not None and not self.bgm_channel.get_busy():
            self.bgm_channel.play(bgm, loops=-1)

    def stop(self):
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        if self.bgm_channel is not None:
            self.bgm_channel.stop()

    def play_hit(self, kind):
        self.play("pop")


class Game:
    def __init__(self):
        pygame.init()

        self.fullscreen = CONFIG["fullscreen"]
        self.screen = self.create_display()

        pygame.display.set_caption(CONFIG["window_title"])

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 40)
        self.small_font = pygame.font.SysFont(None, 28)
        self.big_font = pygame.font.SysFont(None, 70)
        self.huge_font = pygame.font.SysFont(None, 150)

        self.tracker = LaserTracker()
        self.receiver = ShotReceiver()
        self.sound = SoundManager()

        self.balloons = []
        self.pop_bursts = []
        self.score = 0
        self.combo = 0
        self.shot_count = 0

        self.start_time = time.time()
        self.last_spawn = 0

        self.running = True
        self.state = "start"
        self.final_score = 0
        self.current_name = ""
        self.today_ranking = []
        self.last_saved_entry_id = None
        self.ranking_file = Path(__file__).with_name(
            CONFIG.get("daily_ranking_file", "daily_rankings.json")
        )

        self.laser = None

        self.last_shot_pos = None
        self.last_shot_hit = False
        self.last_shot_time = 0
        self.game_over_sound_played = False
        self.action_buttons = self.build_action_buttons()
        self.name_buttons = self.build_name_buttons()
        self.name_action_buttons = self.build_name_action_buttons()

    def create_display(self):
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        return pygame.display.set_mode(
            (CONFIG["screen_w"], CONFIG["screen_h"]),
            flags,
        )

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.screen = self.create_display()
        self.action_buttons = self.build_action_buttons()
        self.name_buttons = self.build_name_buttons()
        self.name_action_buttons = self.build_name_action_buttons()

    def build_action_buttons(self):
        center_x = CONFIG["screen_w"] // 2
        center_y = CONFIG["screen_h"] // 2
        button_w = CONFIG.get("action_button_width", 260)
        button_h = CONFIG.get("action_button_height", 160)

        def make_rect(offset_y):
            return pygame.Rect(
                center_x - button_w // 2,
                center_y + offset_y - button_h // 2,
                button_w,
                button_h,
            )

        return {
            "start": make_rect(40),
            "score": make_rect(80),
            "name": make_rect(150),
            "home": make_rect(180),
        }

    def build_name_buttons(self):
        rows = CONFIG.get(
            "name_keyboard_rows",
            ["ABCDEFG", "HIJKLMN", "OPQRSTU", "VWXYZ-"],
        )
        cols = max(len(row) for row in rows)
        key_w = CONFIG.get("name_key_width", 92)
        key_h = CONFIG.get("name_key_height", 68)
        gap_x = CONFIG.get("name_key_gap_x", 12)
        gap_y = CONFIG.get("name_key_gap_y", 12)
        total_w = cols * key_w + (cols - 1) * gap_x
        start_x = CONFIG["screen_w"] // 2 - total_w // 2
        start_y = CONFIG.get("name_keyboard_y", 260)
        buttons = {}

        for row_index, row in enumerate(rows):
            row_w = len(row) * key_w + max(0, len(row) - 1) * gap_x
            row_x = CONFIG["screen_w"] // 2 - row_w // 2
            for col_index, char in enumerate(row):
                rect = pygame.Rect(
                    row_x + col_index * (key_w + gap_x),
                    start_y + row_index * (key_h + gap_y),
                    key_w,
                    key_h,
                )
                buttons[char] = rect

        return buttons

    def build_name_action_buttons(self):
        labels = ["DEL", "CLEAR", "OK"]
        button_w = CONFIG.get("name_action_width", 160)
        button_h = CONFIG.get("name_action_height", 72)
        gap = CONFIG.get("name_action_gap", 18)
        total_w = len(labels) * button_w + (len(labels) - 1) * gap
        start_x = CONFIG["screen_w"] // 2 - total_w // 2
        y = CONFIG.get("name_action_y", 610)

        return {
            label: pygame.Rect(
                start_x + index * (button_w + gap),
                y,
                button_w,
                button_h,
            )
            for index, label in enumerate(labels)
        }

    def today_key(self):
        return time.strftime("%Y-%m-%d")

    def load_ranking_data(self):
        if not self.ranking_file.exists():
            return {}

        try:
            with self.ranking_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_ranking_data(self, data):
        with self.ranking_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    def get_today_ranking(self):
        data = self.load_ranking_data()
        entries = data.get(self.today_key(), [])
        if not isinstance(entries, list):
            return []

        sorted_entries = sorted(
            entries,
            key=lambda item: (
                -int(item.get("score", 0)),
                float(item.get("created_at", 0)),
            ),
        )
        return sorted_entries[: CONFIG.get("daily_ranking_limit", 10)]

    def save_today_score(self, name):
        clean_name = name.strip().upper()
        if not clean_name:
            return False

        data = self.load_ranking_data()
        today = self.today_key()
        entries = data.get(today, [])
        if not isinstance(entries, list):
            entries = []

        entry_id = str(int(time.time() * 1000))
        entries.append(
            {
                "id": entry_id,
                "name": clean_name,
                "score": int(self.final_score),
                "shots": int(self.shot_count),
                "created_at": time.time(),
            }
        )
        data[today] = entries
        self.save_ranking_data(data)
        self.last_saved_entry_id = entry_id
        self.today_ranking = self.get_today_ranking()
        return True

    def reset_round(self):
        self.balloons = []
        self.pop_bursts = []
        self.score = 0
        self.combo = 0
        self.shot_count = 0
        self.start_time = time.time()
        self.last_spawn = 0
        self.last_shot_pos = None
        self.last_shot_hit = False
        self.last_shot_time = 0
        self.game_over_sound_played = False

    def start_round(self):
        self.reset_round()
        self.final_score = 0
        self.current_name = ""
        self.last_saved_entry_id = None
        self.today_ranking = []
        self.state = "play"

    def finish_round(self):
        self.final_score = self.score
        self.balloons = []
        self.pop_bursts = []
        self.combo = 0
        self.current_name = ""
        self.state = "score_prompt"
        if not self.game_over_sound_played:
            self.sound.play("game_over")
            self.game_over_sound_played = True

    def return_to_start(self):
        self.balloons = []
        self.pop_bursts = []
        self.combo = 0
        self.last_shot_pos = None
        self.last_shot_hit = False
        self.current_name = ""
        self.state = "start"

    def point_hits_button(self, point, button_key):
        if point is None:
            return False
        return self.action_buttons[button_key].collidepoint(int(point[0]), int(point[1]))

    def handle_menu_shot(self, point):
        self.last_shot_pos = point
        self.last_shot_hit = False
        self.last_shot_time = time.time()

        if self.state == "start" and self.point_hits_button(point, "start"):
            self.last_shot_hit = True
            self.start_round()
            return True

        if self.state == "score_prompt" and self.point_hits_button(point, "score"):
            self.last_shot_hit = True
            self.state = "score_view"
            return True

        if self.state == "score_view" and self.point_hits_button(point, "name"):
            self.last_shot_hit = True
            self.state = "name_entry"
            return True

        if self.state == "name_entry":
            handled = self.handle_name_entry_shot(point)
            self.last_shot_hit = handled
            return handled

        if self.state == "ranking_view" and self.point_hits_button(point, "home"):
            self.last_shot_hit = True
            self.return_to_start()
            return True

        return False

    def handle_name_entry_shot(self, point):
        if point is None:
            return False

        max_length = CONFIG.get("player_name_max_length", 6)

        for char, rect in self.name_buttons.items():
            if rect.collidepoint(int(point[0]), int(point[1])):
                if len(self.current_name) < max_length:
                    self.current_name += char
                    return True
                return False

        if self.name_action_buttons["DEL"].collidepoint(int(point[0]), int(point[1])):
            if self.current_name:
                self.current_name = self.current_name[:-1]
                return True
            return False

        if self.name_action_buttons["CLEAR"].collidepoint(int(point[0]), int(point[1])):
            if self.current_name:
                self.current_name = ""
                return True
            return False

        if self.name_action_buttons["OK"].collidepoint(int(point[0]), int(point[1])):
            if self.save_today_score(self.current_name):
                self.state = "ranking_view"
                return True
            return False

        return False

    def choose_balloon_kind(self):
        r = random.random()

        if r < CONFIG["spawn_rate_normal"]:
            return "normal"

        if r < CONFIG["spawn_rate_normal"] + CONFIG["spawn_rate_bonus"]:
            return "bonus"

        return "bomb"

    def spawn_balloon(self):
        if len(self.balloons) >= CONFIG["max_balloons"]:
            return

        x = random.randint(
            CONFIG["spawn_x_margin"],
            CONFIG["screen_w"] - CONFIG["spawn_x_margin"],
        )

        y = CONFIG["screen_h"] + random.randint(
            CONFIG["spawn_y_min"],
            CONFIG["spawn_y_max"],
        )

        radius = random.randint(
            CONFIG["balloon_radius_min"],
            CONFIG["balloon_radius_max"],
        )

        speed = random.uniform(
            CONFIG["balloon_speed_min"],
            CONFIG["balloon_speed_max"],
        )

        kind = self.choose_balloon_kind()

        self.balloons.append(
            Balloon(x, y, radius, speed, kind)
        )

    def balloon_color(self, kind):
        if kind == "normal":
            return CONFIG["balloon_normal_color"]
        if kind == "bonus":
            return CONFIG["balloon_bonus_color"]
        return CONFIG["balloon_bomb_color"]

    def balloon_outline_color(self, kind):
        if kind == "normal":
            return CONFIG["balloon_normal_outline_color"]
        if kind == "bonus":
            return CONFIG["balloon_bonus_outline_color"]
        return CONFIG["balloon_bomb_outline_color"]

    def spawn_pop_burst(self, balloon):
        particles = []
        duration = CONFIG["pop_effect_duration_sec"]

        for _ in range(CONFIG["pop_particle_count"]):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(
                CONFIG["pop_particle_speed_min"],
                CONFIG["pop_particle_speed_max"],
            )
            particles.append(
                PopParticle(
                    x=balloon.x,
                    y=balloon.y,
                    vx=np.cos(angle) * speed,
                    vy=np.sin(angle) * speed,
                    radius=random.uniform(3, 7),
                    color=self.balloon_color(balloon.kind),
                    life=duration,
                )
            )

        self.pop_bursts.append(
            PopBurst(
                x=balloon.x,
                y=balloon.y,
                radius=balloon.radius,
                color=self.balloon_color(balloon.kind),
                life=duration,
                max_life=duration,
                particles=particles,
            )
        )

    def multiplier(self):
        if self.combo >= CONFIG["combo_x3_count"]:
            return CONFIG["combo_x3_multiplier"]

        if self.combo >= CONFIG["combo_x2_count"]:
            return CONFIG["combo_x2_multiplier"]

        return 1

    def balloon_hit_center(self, balloon):
        # Compensate for camera/projector latency by pulling the hit area
        # slightly back toward the balloon position the player still sees.
        delay_sec = float(CONFIG.get("balloon_hit_compensation_sec", 0.0))
        offset_y = float(CONFIG.get("balloon_hit_offset_y", 0.0))
        return balloon.x, balloon.y + balloon.speed * delay_sec + offset_y

    def hit_test(self, point):
        self.last_shot_pos = point
        self.last_shot_hit = False
        self.last_shot_time = time.time()

        if point is None:
            self.combo = 0
            self.sound.play("miss")
            return

        px, py = point

        for balloon in reversed(self.balloons):

            if not balloon.alive:
                continue

            hit_x, hit_y = self.balloon_hit_center(balloon)
            dx = px - hit_x
            dy = py - hit_y

            if dx * dx + dy * dy <= balloon.radius * balloon.radius:

                balloon.alive = False
                self.spawn_pop_burst(balloon)

                gained = balloon.score()

                if gained > 0:
                    self.combo += 1
                    self.score += gained * self.multiplier()
                else:
                    self.combo = 0
                    self.score += gained

                self.sound.play_hit(balloon.kind)
                self.last_shot_hit = True
                return

        self.combo = 0
        self.sound.play("miss")

    def update_play(self, dt):

        elapsed = time.time() - self.start_time

        if elapsed >= CONFIG["game_time_sec"]:
            self.finish_round()
            return

        if elapsed - self.last_spawn >= CONFIG["spawn_interval"]:
            self.spawn_balloon()
            self.last_spawn = elapsed

        for b in self.balloons:
            b.update(dt)

        for burst in self.pop_bursts:
            burst.update(dt)

        self.balloons = [b for b in self.balloons if b.alive]
        self.pop_bursts = [burst for burst in self.pop_bursts if burst.alive()]

        if self.receiver.poll():
            self.sound.play("shot")
            self.shot_count += 1
            shot_point = self.tracker.read_for_shot()
            self.laser = shot_point
            self.hit_test(shot_point)
            self.tracker.set_shot_feedback(self.last_shot_hit)

    def update_menu(self):
        if self.receiver.poll():
            self.sound.play("shot")
            shot_point = self.tracker.read_for_shot()
            self.laser = shot_point
            hit = self.handle_menu_shot(shot_point)
            self.tracker.set_shot_feedback(hit)

    def update(self, dt):
        self.laser = self.tracker.read()

        if self.state == "play":
            self.update_play(dt)
            return

        self.update_menu()

    def draw_balloon(self, b):
        color = self.balloon_color(b.kind)
        outline = self.balloon_outline_color(b.kind)

        pygame.draw.circle(
            self.screen,
            color,
            (int(b.x), int(b.y)),
            b.radius,
        )

        pygame.draw.circle(
            self.screen,
            outline,
            (int(b.x), int(b.y)),
            b.radius,
            CONFIG["balloon_outline_width"],
        )

        pygame.draw.line(
            self.screen,
            CONFIG["balloon_string_color"],
            (int(b.x), int(b.y + b.radius)),
            (int(b.x), int(b.y + b.radius + 30)),
            2,
        )

    def draw_pop_burst(self, burst):
        progress = 1 - max(0, burst.life) / burst.max_life
        ring_radius = int(burst.radius * (0.6 + progress * 1.2))
        ring_width = max(1, int(6 * (1 - progress)))

        pygame.draw.circle(
            self.screen,
            burst.color,
            (int(burst.x), int(burst.y)),
            max(1, ring_radius),
            ring_width,
        )

        flash_radius = max(2, int(burst.radius * (1 - progress * 0.7)))
        pygame.draw.circle(
            self.screen,
            CONFIG["pop_flash_color"],
            (int(burst.x), int(burst.y)),
            flash_radius,
            1,
        )

        for particle in burst.particles:
            particle_progress = max(0.15, particle.life / burst.max_life)
            particle_radius = max(1, int(particle.radius * particle_progress))
            pygame.draw.circle(
                self.screen,
                particle.color,
                (int(particle.x), int(particle.y)),
                particle_radius,
            )

    def draw_shot_marker(self):

        if not CONFIG["show_shot_marker"]:
            return

        if self.last_shot_pos is None:
            return

        if time.time() - self.last_shot_time > CONFIG["shot_marker_duration_sec"]:
            return

        x, y = self.last_shot_pos

        if x is None or y is None:
            return

        color = (
            CONFIG["shot_hit_color"]
            if self.last_shot_hit
            else CONFIG["shot_miss_color"]
        )

        r = CONFIG["shot_marker_radius"]

        pygame.draw.circle(self.screen, color, (int(x), int(y)), r, 3)

        pygame.draw.line(
            self.screen,
            color,
            (int(x) - r, int(y)),
            (int(x) + r, int(y)),
            2,
        )

        pygame.draw.line(
            self.screen,
            color,
            (int(x), int(y) - r),
            (int(x), int(y) + r),
            2,
        )

    def shot_status_text(self):
        active_window = CONFIG.get(
            "shot_status_duration_sec",
            CONFIG.get("shot_cooldown_sec", 0.15),
        )
        if time.time() - self.receiver.last_shot <= active_window:
            return "SHOT: ON"
        return "SHOT: OFF"

    def draw_status_panel(self, remain):
        panel_x = CONFIG.get("ui_panel_x", 16)
        panel_y = CONFIG.get("ui_panel_y", 16)
        panel_w = CONFIG.get("ui_panel_width", 220)
        panel_h = CONFIG.get("ui_panel_height", 118)

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(CONFIG.get("ui_panel_color", (8, 12, 18, 150)))
        self.screen.blit(panel, (panel_x, panel_y))

        line_gap = CONFIG.get("ui_line_gap", 34)
        label_color = CONFIG.get("ui_text_color", CONFIG["text_color"])
        shot_active = self.shot_status_text() == "SHOT: ON"
        shot_color = CONFIG.get(
            "ui_shot_on_color" if shot_active else "ui_shot_off_color",
            label_color,
        )

        lines = [
            (f"Score {self.score}", label_color),
            (f"Time  {remain}", label_color),
            (self.shot_status_text(), shot_color),
        ]

        for index, (text, color) in enumerate(lines):
            self.screen.blit(
                self.font.render(text, True, color),
                (panel_x + 16, panel_y + 12 + index * line_gap),
            )

    def draw_button_box(self, rect, fill_color=None, outline_color=None, border_radius=20):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(fill_color or CONFIG.get("action_button_color", (10, 14, 22, 185)))
        self.screen.blit(panel, rect.topleft)

        pygame.draw.rect(
            self.screen,
            outline_color or CONFIG.get("action_button_outline_color", (78, 96, 112)),
            rect,
            2,
            border_radius=border_radius,
        )

    def draw_action_button(self, rect, icon, label):
        self.draw_button_box(rect, border_radius=26)

        cx, cy = rect.center
        icon_color = CONFIG.get("action_icon_color", (120, 138, 150))

        if icon == "play":
            points = [
                (cx - 24, cy - 36),
                (cx - 24, cy + 36),
                (cx + 42, cy),
            ]
            pygame.draw.polygon(self.screen, icon_color, points)
        elif icon == "score":
            bar_width = 24
            gaps = 10
            heights = [36, 56, 82]
            left = cx - (bar_width * 3 + gaps * 2) // 2
            base_y = cy + 42
            for index, height in enumerate(heights):
                bar_rect = pygame.Rect(
                    left + index * (bar_width + gaps),
                    base_y - height,
                    bar_width,
                    height,
                )
                pygame.draw.rect(self.screen, icon_color, bar_rect, border_radius=8)
        elif icon == "home":
            roof = [
                (cx, cy - 48),
                (cx - 54, cy - 2),
                (cx + 54, cy - 2),
            ]
            body = pygame.Rect(cx - 38, cy - 2, 76, 62)
            door = pygame.Rect(cx - 12, cy + 22, 24, 38)
            pygame.draw.polygon(self.screen, icon_color, roof)
            pygame.draw.rect(self.screen, icon_color, body, border_radius=10)
            pygame.draw.rect(
                self.screen,
                CONFIG["bg_color"],
                door,
                border_radius=8,
            )
        elif icon == "name":
            tag_rect = pygame.Rect(cx - 48, cy - 30, 96, 56)
            pygame.draw.rect(self.screen, icon_color, tag_rect, 2, border_radius=12)
            dot = self.small_font.render("Aa", True, icon_color)
            self.screen.blit(
                dot,
                (cx - dot.get_width() // 2, cy - dot.get_height() // 2 - 4),
            )

        label_surface = self.small_font.render(
            label,
            True,
            CONFIG.get("action_label_color", CONFIG["text_color"]),
        )
        self.screen.blit(
            label_surface,
            (
                rect.centerx - label_surface.get_width() // 2,
                rect.bottom - 34,
            ),
        )

    def draw_center_message(self, title, subtitle=None, title_color=None, subtitle_color=None):
        title_surface = self.big_font.render(
            title,
            True,
            title_color or CONFIG["text_color"],
        )
        self.screen.blit(
            title_surface,
            (
                CONFIG["screen_w"] // 2 - title_surface.get_width() // 2,
                110,
            ),
        )

        if subtitle:
            subtitle_surface = self.small_font.render(
                subtitle,
                True,
                subtitle_color or CONFIG.get("ui_text_color", CONFIG["text_color"]),
            )
            self.screen.blit(
                subtitle_surface,
                (
                    CONFIG["screen_w"] // 2 - subtitle_surface.get_width() // 2,
                    185,
                ),
            )

    def draw_name_entry_screen(self):
        self.draw_center_message(
            "ENTER NAME",
            "Shoot letters, then OK",
            title_color=CONFIG.get("name_title_color", CONFIG["text_color"]),
            subtitle_color=CONFIG.get("name_subtitle_color", CONFIG.get("ui_text_color")),
        )

        field_rect = pygame.Rect(
            CONFIG["screen_w"] // 2 - 260,
            180,
            520,
            64,
        )
        self.draw_button_box(
            field_rect,
            fill_color=CONFIG.get("name_field_color", (8, 12, 18, 180)),
            outline_color=CONFIG.get("name_field_outline_color", (68, 82, 94)),
            border_radius=18,
        )

        shown_name = self.current_name or "_" * CONFIG.get("player_name_min_length_hint", 3)
        name_surface = self.big_font.render(
            shown_name,
            True,
            CONFIG.get("name_value_color", (120, 136, 150)),
        )
        self.screen.blit(
            name_surface,
            (
                field_rect.centerx - name_surface.get_width() // 2,
                field_rect.centery - name_surface.get_height() // 2,
            ),
        )

        hint = self.small_font.render(
            f"MAX {CONFIG.get('player_name_max_length', 6)}",
            True,
            CONFIG.get("name_hint_color", (84, 96, 108)),
        )
        self.screen.blit(
            hint,
            (
                field_rect.right + 18,
                field_rect.centery - hint.get_height() // 2,
            ),
        )

        for char, rect in self.name_buttons.items():
            self.draw_button_box(
                rect,
                fill_color=CONFIG.get("name_key_color", (8, 12, 18, 170)),
                outline_color=CONFIG.get("name_key_outline_color", (64, 78, 90)),
                border_radius=14,
            )
            label_surface = self.font.render(
                char,
                True,
                CONFIG.get("name_key_text_color", (116, 130, 142)),
            )
            self.screen.blit(
                label_surface,
                (
                    rect.centerx - label_surface.get_width() // 2,
                    rect.centery - label_surface.get_height() // 2,
                ),
            )

        for action, rect in self.name_action_buttons.items():
            self.draw_button_box(
                rect,
                fill_color=CONFIG.get("name_action_color", (10, 15, 22, 185)),
                outline_color=CONFIG.get("name_action_outline_color", (74, 88, 100)),
                border_radius=16,
            )
            label_surface = self.small_font.render(
                action,
                True,
                CONFIG.get("name_action_text_color", (118, 132, 144)),
            )
            self.screen.blit(
                label_surface,
                (
                    rect.centerx - label_surface.get_width() // 2,
                    rect.centery - label_surface.get_height() // 2,
                ),
            )

    def draw_ranking_screen(self):
        self.today_ranking = self.get_today_ranking()
        self.draw_center_message(
            "TODAY RANKING",
            self.today_key(),
            title_color=CONFIG.get("ranking_title_color", CONFIG["text_color"]),
            subtitle_color=CONFIG.get("ranking_subtitle_color", CONFIG.get("ui_text_color")),
        )

        display_count = CONFIG.get("daily_ranking_display_count", 10)
        board_rect = pygame.Rect(
            CONFIG["screen_w"] // 2 - 300,
            210,
            600,
            CONFIG.get("ranking_board_height", 360),
        )
        self.draw_button_box(
            board_rect,
            fill_color=CONFIG.get("ranking_panel_color", (7, 11, 17, 180)),
            outline_color=CONFIG.get("ranking_panel_outline_color", (60, 72, 84)),
            border_radius=22,
        )

        entries = self.today_ranking[:display_count]
        row_gap = 6
        top_padding = 18
        side_padding = 18
        usable_rows = max(1, display_count)
        usable_height = board_rect.height - top_padding * 2 - row_gap * (usable_rows - 1)
        row_h = max(24, usable_height // usable_rows)
        row_y = board_rect.y + top_padding
        row_font = self.small_font if display_count >= 8 else self.font
        rank_font = self.small_font

        if not entries:
            empty_surface = self.font.render(
                "NO SCORES YET",
                True,
                CONFIG.get("ranking_empty_color", (86, 98, 110)),
            )
            self.screen.blit(
                empty_surface,
                (
                    board_rect.centerx - empty_surface.get_width() // 2,
                    board_rect.centery - empty_surface.get_height() // 2,
                ),
            )
        else:
            for index, entry in enumerate(entries, start=1):
                row_rect = pygame.Rect(
                    board_rect.x + side_padding,
                    row_y,
                    board_rect.width - side_padding * 2,
                    row_h,
                )
                is_player = entry.get("id") == self.last_saved_entry_id
                self.draw_button_box(
                    row_rect,
                    fill_color=CONFIG.get(
                        "ranking_highlight_color" if is_player else "ranking_row_color",
                        (12, 18, 26, 185) if is_player else (8, 12, 18, 120),
                    ),
                    outline_color=CONFIG.get(
                        "ranking_highlight_outline_color" if is_player else "ranking_row_outline_color",
                        (92, 108, 122) if is_player else (50, 62, 74),
                    ),
                    border_radius=14,
                )

                rank_surface = rank_font.render(
                    f"{index}",
                    True,
                    CONFIG.get("ranking_rank_color", (112, 126, 138)),
                )
                name_surface = row_font.render(
                    entry.get("name", "---"),
                    True,
                    CONFIG.get("ranking_name_color", (122, 138, 150)),
                )
                score_surface = row_font.render(
                    str(entry.get("score", 0)),
                    True,
                    CONFIG.get("ranking_score_color", (132, 146, 158)),
                )

                self.screen.blit(
                    rank_surface,
                    (row_rect.x + 18, row_rect.centery - rank_surface.get_height() // 2),
                )
                self.screen.blit(
                    name_surface,
                    (row_rect.x + 64, row_rect.centery - name_surface.get_height() // 2),
                )
                self.screen.blit(
                    score_surface,
                    (
                        row_rect.right - score_surface.get_width() - 22,
                        row_rect.centery - score_surface.get_height() // 2,
                    ),
                )
                row_y += row_h + row_gap

        self.draw_action_button(self.action_buttons["home"], "home", "START")

    def draw_start_screen(self):
        self.draw_center_message(
            "LASER BALLOON",
            "Shoot the play mark to start",
        )
        self.draw_action_button(self.action_buttons["start"], "play", "START")

    def draw_score_prompt_screen(self):
        self.draw_center_message(
            "TIME UP",
            "Shoot the score mark",
        )
        self.draw_action_button(self.action_buttons["score"], "score", "SHOW SCORE")

    def draw_score_view_screen(self):
        self.draw_center_message(
            "YOUR SCORE",
            "Shoot the name mark",
            title_color=CONFIG.get("score_title_color", (56, 66, 76)),
            subtitle_color=CONFIG.get("score_subtitle_color", (68, 80, 92)),
        )

        score_surface = self.huge_font.render(
            str(self.final_score),
            True,
            CONFIG.get("score_value_color", (180, 198, 210)),
        )
        self.screen.blit(
            score_surface,
            (
                CONFIG["screen_w"] // 2 - score_surface.get_width() // 2,
                250,
            ),
        )

        self.draw_action_button(self.action_buttons["name"], "name", "NAME")

    def draw(self):

        self.screen.fill(CONFIG["bg_color"])

        if self.state == "play":
            for b in self.balloons:
                self.draw_balloon(b)

            for burst in self.pop_bursts:
                self.draw_pop_burst(burst)

        if CONFIG.get("show_projected_laser_cursor", False) and self.laser is not None:

            lx, ly = self.laser

            if 0 <= lx < CONFIG["screen_w"] and 0 <= ly < CONFIG["screen_h"]:
                pygame.draw.circle(
                    self.screen,
                    CONFIG["laser_cursor_color"],
                    (int(lx), int(ly)),
                    8,
                    2,
                )

        self.draw_shot_marker()

        projector_dim_alpha = CONFIG.get("projector_dim_alpha", 0)
        if projector_dim_alpha > 0:
            dim_overlay = pygame.Surface(
                (CONFIG["screen_w"], CONFIG["screen_h"]),
                pygame.SRCALPHA,
            )
            dim_overlay.fill((0, 0, 0, projector_dim_alpha))
            self.screen.blit(dim_overlay, (0, 0))

        remain = max(
            0,
            CONFIG["game_time_sec"] - int(time.time() - self.start_time),
        )

        if self.state == "play" and CONFIG.get("show_ui_text", True):
            self.draw_status_panel(remain)
        elif self.state == "start":
            self.draw_start_screen()
        elif self.state == "score_prompt":
            self.draw_score_prompt_screen()
        elif self.state == "score_view":
            self.draw_score_view_screen()
        elif self.state == "name_entry":
            self.draw_name_entry_screen()
        elif self.state == "ranking_view":
            self.draw_ranking_screen()

        pygame.display.flip()

    def run(self):

        while self.running:

            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    aim_step = 10 if mods & pygame.KMOD_SHIFT else 2
                    cam_step = (
                        CONFIG.get("cam_point_step_fast", 10)
                        if mods & pygame.KMOD_SHIFT
                        else CONFIG.get("cam_point_step", 2)
                    )

                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_F11:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_c:
                        self.tracker.toggle_calibration_mode()
                    if event.key == pygame.K_TAB:
                        self.tracker.select_next_cam_point()
                    if event.key == pygame.K_a:
                        self.tracker.auto_calibrate()
                    if event.key in (pygame.K_1, pygame.K_KP1):
                        self.tracker.select_cam_point(0)
                    if event.key in (pygame.K_2, pygame.K_KP2):
                        self.tracker.select_cam_point(1)
                    if event.key in (pygame.K_3, pygame.K_KP3):
                        self.tracker.select_cam_point(2)
                    if event.key in (pygame.K_4, pygame.K_KP4):
                        self.tracker.select_cam_point(3)
                    if mods & pygame.KMOD_CTRL:
                        if event.key == pygame.K_LEFT:
                            self.tracker.adjust_screen_offset(-aim_step, 0)
                        if event.key == pygame.K_RIGHT:
                            self.tracker.adjust_screen_offset(aim_step, 0)
                        if event.key == pygame.K_UP:
                            self.tracker.adjust_screen_offset(0, -aim_step)
                        if event.key == pygame.K_DOWN:
                            self.tracker.adjust_screen_offset(0, aim_step)
                        if event.key == pygame.K_s:
                            self.tracker.save_calibration_to_config()
                    elif self.tracker.calibration_mode:
                        if event.key == pygame.K_LEFT:
                            self.tracker.move_selected_cam_point(-cam_step, 0)
                        if event.key == pygame.K_RIGHT:
                            self.tracker.move_selected_cam_point(cam_step, 0)
                        if event.key == pygame.K_UP:
                            self.tracker.move_selected_cam_point(0, -cam_step)
                        if event.key == pygame.K_DOWN:
                            self.tracker.move_selected_cam_point(0, cam_step)

            self.update(dt)
            self.draw()

        self.receiver.release()
        self.tracker.release()
        self.sound.stop()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
