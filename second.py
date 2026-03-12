import cv2
import numpy as np
import pygame
import serial
import time
import random
from dataclasses import dataclass
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
        self.cap = cv2.VideoCapture(CONFIG["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.get("camera_w", 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.get("camera_h", 720))
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

        screen_points = np.array(
            [
                [0, 0],
                [CONFIG["screen_w"], 0],
                [CONFIG["screen_w"], CONFIG["screen_h"]],
                [0, CONFIG["screen_h"]],
            ],
            dtype=np.float32,
        )

        self.h_matrix = cv2.getPerspectiveTransform(self.cam_points, screen_points)

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

        for index, (x, y) in enumerate(points):
            cv2.circle(overlay, (x, y), 6, (0, 200, 255), -1)
            cv2.putText(
                overlay,
                f"P{index + 1} ({x}, {y})",
                (x + 8, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
            )

        cv2.putText(
            overlay,
            "Adjust camera and cam_points with this frame",
            (20, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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

        return overlay

    def set_shot_feedback(self, hit):
        self.feedback_text = "HIT" if hit else "NO HIT"
        self.feedback_color = (
            CONFIG["shot_hit_color"] if hit else CONFIG["shot_miss_color"]
        )
        self.feedback_until = time.time() + CONFIG["camera_feedback_duration_sec"]
        self.feedback_camera_point = self.last_camera_point

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        camera_view = self.draw_calibration_overlay(frame.copy())
        tracking_view = camera_view.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray,
            CONFIG["laser_threshold"],
            255,
            cv2.THRESH_BINARY,
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        best = None
        best_area = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if CONFIG["laser_min_area"] <= area <= CONFIG["laser_max_area"]:
                if area > best_area:
                    best_area = area
                    best = centroids[i]

        screen_point = None
        self.last_camera_point = None

        if best is not None:
            cx, cy = best
            self.last_camera_point = (int(cx), int(cy))
            src = np.array([[[cx, cy]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.h_matrix)
            sx, sy = dst[0][0]
            screen_point = (sx, sy)

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
                f"screen: ({int(sx)}, {int(sy)})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            cv2.putText(
                tracking_view,
                "laser: not found",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
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
            cv2.imshow("tracking", tracking_view)
            cv2.waitKey(1)

        return screen_point

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


class Game:
    def __init__(self):
        pygame.init()

        flags = pygame.FULLSCREEN if CONFIG["fullscreen"] else 0

        self.screen = pygame.display.set_mode(
            (CONFIG["screen_w"], CONFIG["screen_h"]),
            flags,
        )

        pygame.display.set_caption(CONFIG["window_title"])

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 40)
        self.small_font = pygame.font.SysFont(None, 28)
        self.big_font = pygame.font.SysFont(None, 70)

        self.tracker = LaserTracker()
        self.receiver = ShotReceiver()

        self.balloons = []
        self.pop_bursts = []
        self.score = 0
        self.combo = 0
        self.shot_count = 0

        self.start_time = time.time()
        self.last_spawn = 0

        self.running = True
        self.game_over = False

        self.laser = None

        self.last_shot_pos = None
        self.last_shot_hit = False
        self.last_shot_time = 0

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

    def hit_test(self, point):
        self.last_shot_pos = point
        self.last_shot_hit = False
        self.last_shot_time = time.time()

        if point is None:
            self.combo = 0
            return

        px, py = point

        for balloon in reversed(self.balloons):

            if not balloon.alive:
                continue

            dx = px - balloon.x
            dy = py - balloon.y

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

                self.last_shot_hit = True
                return

        self.combo = 0

    def update(self, dt):

        if self.game_over:
            return

        elapsed = time.time() - self.start_time

        if elapsed >= CONFIG["game_time_sec"]:
            self.game_over = True
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

        self.laser = self.tracker.read()

        if self.receiver.poll():
            self.shot_count += 1
            self.hit_test(self.laser)
            self.tracker.set_shot_feedback(self.last_shot_hit)

    def draw_balloon(self, b):
        color = self.balloon_color(b.kind)

        pygame.draw.circle(
            self.screen,
            color,
            (int(b.x), int(b.y)),
            b.radius,
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
            (255, 255, 255),
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

    def draw(self):

        self.screen.fill(CONFIG["bg_color"])

        for b in self.balloons:
            self.draw_balloon(b)

        for burst in self.pop_bursts:
            self.draw_pop_burst(burst)

        if self.laser is not None:

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

        remain = max(
            0,
            CONFIG["game_time_sec"] - int(time.time() - self.start_time),
        )

        self.screen.blit(
            self.font.render(f"Score: {self.score}", True, CONFIG["text_color"]),
            (20, 20),
        )

        self.screen.blit(
            self.font.render(f"Combo: {self.combo}", True, CONFIG["text_color"]),
            (20, 60),
        )

        self.screen.blit(
            self.font.render(f"Time: {remain}", True, CONFIG["text_color"]),
            (20, 100),
        )

        self.screen.blit(
            self.font.render(
                f"Shots: {self.shot_count}",
                True,
                CONFIG["text_color"],
            ),
            (20, 140),
        )

        for index, line in enumerate(self.receiver.debug_lines()):
            self.screen.blit(
                self.small_font.render(line, True, CONFIG["text_color"]),
                (20, 185 + index * 28),
            )

        if self.game_over:

            txt = self.big_font.render(
                "GAME OVER",
                True,
                CONFIG["text_color"],
            )

            self.screen.blit(
                txt,
                (
                    CONFIG["screen_w"] // 2 - txt.get_width() // 2,
                    CONFIG["screen_h"] // 2 - 80,
                ),
            )

        pygame.display.flip()

    def run(self):

        while self.running:

            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            self.update(dt)
            self.draw()

        self.receiver.release()
        self.tracker.release()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
