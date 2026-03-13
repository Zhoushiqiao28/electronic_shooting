import cv2
import numpy as np
import pygame

from config import CONFIG


TARGET_RADIUS = 26
SAMPLE_RADIUS = 4
TRACKING_WINDOW_SCALE = 0.55


def open_camera():
    backend_name = str(CONFIG.get("camera_backend", "any")).lower()
    if backend_name == "any":
        backends = ["any", "dshow", "msmf"]
    elif backend_name == "dshow":
        backends = ["dshow", "any", "msmf"]
    elif backend_name == "msmf":
        backends = ["msmf", "any", "dshow"]
    else:
        backends = ["any", "dshow", "msmf"]

    backend_ids = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    indices = [CONFIG.get("camera_index", 0)]
    for index in CONFIG.get("camera_probe_indices", []):
        if index not in indices:
            indices.append(index)

    for backend in backends:
        backend_id = backend_ids[backend]
        for index in indices:
            cap = cv2.VideoCapture(index, backend_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.get("camera_w", 1280))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.get("camera_h", 720))
                ok = False
                for _ in range(15):
                    ok, _ = cap.read()
                    if ok:
                        print(f"Camera opened: index={index}, backend={backend}")
                        return cap
                cap.release()

    raise RuntimeError("Could not open any configured camera")


def scale_cam_points(frame_w, frame_h):
    cam_points = np.array(CONFIG["cam_points"], dtype=np.float32)
    ref_w = CONFIG.get("cam_points_ref_w", frame_w)
    ref_h = CONFIG.get("cam_points_ref_h", frame_h)
    scaled = cam_points.copy()
    if ref_w > 0 and ref_h > 0:
        scaled[:, 0] *= frame_w / ref_w
        scaled[:, 1] *= frame_h / ref_h
    return scaled


def build_roi_mask(frame_w, frame_h, cam_points):
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cam_points.astype(np.int32), 255)
    return mask


def target_points():
    w = CONFIG["screen_w"]
    h = CONFIG["screen_h"]
    mx = int(w * 0.18)
    my = int(h * 0.18)
    return [
        ("Top Left", (mx, my)),
        ("Top Right", (w - mx, my)),
        ("Center", (w // 2, h // 2)),
        ("Bottom Left", (mx, h - my)),
        ("Bottom Right", (w - mx, h - my)),
    ]


def detect_bright_spot(frame, roi_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, roi_mask)

    column_profile = masked_gray.mean(axis=0, keepdims=True).astype(np.float32)
    smooth_profile = cv2.GaussianBlur(column_profile, (81, 1), 0)
    column_bias = smooth_profile - smooth_profile.min()
    bias_image = np.repeat(
        np.clip(column_bias * 0.8, 0, 255).astype(np.uint8),
        masked_gray.shape[0],
        axis=0,
    )
    corrected = cv2.subtract(masked_gray, bias_image)

    local_bg = cv2.GaussianBlur(corrected, (7, 7), 0)
    local_peak = cv2.subtract(corrected, local_bg)
    response = cv2.max(local_peak, corrected)
    response = cv2.GaussianBlur(response, (3, 3), 0)
    response = cv2.bitwise_and(response, roi_mask)

    _, peak_score, _, peak_loc = cv2.minMaxLoc(response, mask=roi_mask)
    if peak_score < CONFIG.get("laser_response_min", 6):
        return None, response

    px, py = peak_loc
    x0 = max(0, px - SAMPLE_RADIUS)
    y0 = max(0, py - SAMPLE_RADIUS)
    x1 = min(frame.shape[1], px + SAMPLE_RADIUS + 1)
    y1 = min(frame.shape[0], py + SAMPLE_RADIUS + 1)

    patch = response[y0:y1, x0:x1].astype(np.float32)
    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float32),
        np.arange(y0, y1, dtype=np.float32),
    )
    total = float(patch.sum())

    if total > 0:
        cx = float((patch * grid_x).sum() / total)
        cy = float((patch * grid_y).sum() / total)
    else:
        cx = float(px)
        cy = float(py)

    return (cx, cy, peak_score), response


def sample_hsv(frame, point):
    cx, cy = int(point[0]), int(point[1])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x0 = max(0, cx - SAMPLE_RADIUS)
    y0 = max(0, cy - SAMPLE_RADIUS)
    x1 = min(frame.shape[1], cx + SAMPLE_RADIUS + 1)
    y1 = min(frame.shape[0], cy + SAMPLE_RADIUS + 1)
    patch = hsv[y0:y1, x0:x1]
    return {
        "h_mean": float(patch[:, :, 0].mean()),
        "s_mean": float(patch[:, :, 1].mean()),
        "v_mean": float(patch[:, :, 2].mean()),
        "h_min": int(patch[:, :, 0].min()),
        "h_max": int(patch[:, :, 0].max()),
        "s_min": int(patch[:, :, 1].min()),
        "s_max": int(patch[:, :, 1].max()),
        "v_min": int(patch[:, :, 2].min()),
        "v_max": int(patch[:, :, 2].max()),
    }


def summarize_samples(samples):
    h_values = []
    s_values = []
    v_values = []

    for sample in samples:
        h_values.extend([sample["h_min"], sample["h_max"], int(sample["h_mean"])])
        s_values.extend([sample["s_min"], sample["s_max"], int(sample["s_mean"])])
        v_values.extend([sample["v_min"], sample["v_max"], int(sample["v_mean"])])

    h_values = np.array(h_values)
    s_values = np.array(s_values)
    v_values = np.array(v_values)

    return {
        "h_low": int(np.percentile(h_values, 5)),
        "h_high": int(np.percentile(h_values, 95)),
        "s_low": int(np.percentile(s_values, 10)),
        "s_high": int(np.percentile(s_values, 95)),
        "v_low": int(np.percentile(v_values, 10)),
        "v_high": int(np.percentile(v_values, 100)),
    }


def draw_screen(screen, font, big_font, targets, current_index, samples, message):
    screen.fill((8, 10, 18))

    for index, (_, point) in enumerate(targets):
        color = (120, 230, 240) if index == current_index else (60, 90, 110)
        width = 4 if index == current_index else 2
        pygame.draw.circle(screen, color, point, TARGET_RADIUS, width)
        pygame.draw.line(screen, color, (point[0] - 14, point[1]), (point[0] + 14, point[1]), 2)
        pygame.draw.line(screen, color, (point[0], point[1] - 14), (point[0], point[1] + 14), 2)

    title = big_font.render("HSV Calibration", True, (180, 210, 235))
    screen.blit(title, (30, 24))

    info_lines = [
        "Move the laser onto the highlighted target and press SPACE.",
        "ENTER also samples. ESC exits.",
        f"Samples: {len(samples)} / {len(targets)}",
        message,
    ]
    for idx, line in enumerate(info_lines):
        screen.blit(font.render(line, True, (170, 190, 210)), (30, 110 + idx * 32))

    if current_index < len(targets):
        name, _ = targets[current_index]
        screen.blit(font.render(f"Current target: {name}", True, (120, 230, 240)), (30, 250))

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((CONFIG["screen_w"], CONFIG["screen_h"]))
    pygame.display.set_caption("HSV Calibration")
    font = pygame.font.SysFont(None, 34)
    big_font = pygame.font.SysFont(None, 56)
    clock = pygame.time.Clock()

    cap = open_camera()
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CONFIG.get("camera_w", 1280)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CONFIG.get("camera_h", 720)
    cam_points = scale_cam_points(frame_w, frame_h)
    roi_mask = build_roi_mask(frame_w, frame_h, cam_points)

    targets = target_points()
    current_index = 0
    samples = []
    message = "Ready"
    last_frame = None
    last_tracking = None
    completed = False
    running = True

    while running:
        clock.tick(60)
        ok, frame = cap.read()
        if ok:
            last_frame = frame.copy()
            detection, response = detect_bright_spot(frame, roi_mask)
            tracking = frame.copy()
            cv2.polylines(tracking, [cam_points.astype(np.int32)], True, (255, 200, 0), 2)

            if detection is not None:
                cx, cy, score = detection
                cv2.circle(tracking, (int(cx), int(cy)), 10, (0, 255, 255), 2)
                cv2.putText(
                    tracking,
                    f"spot ({int(cx)}, {int(cy)}) score:{int(score)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    tracking,
                    "spot: not found",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.namedWindow("hsv_tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "hsv_tracking",
                max(320, int(frame_w * TRACKING_WINDOW_SCALE)),
                max(180, int(frame_h * TRACKING_WINDOW_SCALE)),
            )
            cv2.imshow("hsv_tracking", tracking)
            cv2.imshow("hsv_response", response)
            cv2.waitKey(1)
            last_tracking = detection

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if completed:
                        message = "Sampling already completed. Press ESC to exit."
                        continue

                    if last_frame is None or last_tracking is None:
                        message = "No bright spot found. Try again."
                        continue

                    sample = sample_hsv(last_frame, last_tracking)
                    target_name = targets[current_index][0]
                    samples.append(sample)
                    message = (
                        f"{target_name}: H {sample['h_mean']:.1f} "
                        f"S {sample['s_mean']:.1f} V {sample['v_mean']:.1f}"
                    )
                    current_index += 1

                    if current_index >= len(targets):
                        summary = summarize_samples(samples)
                        print("HSV samples:")
                        for index, sample in enumerate(samples, start=1):
                            print(index, sample)
                        print("Suggested range:", summary)
                        print(
                            "Config snippet:",
                            {
                                "laser_red_h1_min": summary["h_low"],
                                "laser_red_h1_max": summary["h_high"],
                                "laser_s_min": summary["s_low"],
                                "laser_s_max": summary["s_high"],
                                "laser_v_min": summary["v_low"],
                            },
                        )
                        message = (
                            f"Done. Suggested H:{summary['h_low']}-{summary['h_high']} "
                            f"S:{summary['s_low']}-{summary['s_high']} "
                            f"V:{summary['v_low']}-{summary['v_high']}"
                        )
                        current_index = len(targets) - 1
                        completed = True

        draw_screen(screen, font, big_font, targets, current_index, samples, message)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
