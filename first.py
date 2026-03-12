import cv2
import serial
import time
import math

# ===== 設定 =====
SERIAL_PORT = "COM9"   # 自分の環境に合わせて変更
BAUDRATE = 115200
CAMERA_INDEX = 0

# 的の中心座標（最初は仮）
TARGET_CENTER = (320, 240)

# 得点リング半径
RINGS = [
    (40, 10),
    (80, 8),
    (120, 6),
    (160, 4),
    (220, 2),
]

# ===== シリアル接続 =====
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
    time.sleep(2)
except Exception as e:
    print(f"シリアル接続失敗: {e}")
    print("SERIAL_PORT を正しい COM 番号に変更してください。")
    exit()

# ===== カメラ開始 =====
# ===== カメラ開始 =====
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("カメラを開けませんでした")
    ser.close()
    exit()

# 露出まわりを調整
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 80)

print("AUTO_EXPOSURE =", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print("EXPOSURE      =", cap.get(cv2.CAP_PROP_EXPOSURE))
print("GAIN          =", cap.get(cv2.CAP_PROP_GAIN))
print("BRIGHTNESS    =", cap.get(cv2.CAP_PROP_BRIGHTNESS))

last_score = None
total_score = 0
shot_count = 0

status_text = ""
status_color = (255, 255, 255)
status_until = 0

last_hit_point = None

def set_status(text, color=(255, 255, 255), duration=1.0):
    global status_text, status_color, status_until
    status_text = text
    status_color = color
    status_until = time.time() + duration

def detect_red_laser(frame):
    """
    赤いレーザー点を検出して (x, y) を返す。
    見つからなければ None
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 赤色はHSV空間で両端に分かれる
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 100, 100)
    upper_red2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # ノイズ軽減
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 小さい点を少し拾いやすくする
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 一番大きい赤領域を採用
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5:
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def calc_score(point, center):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    dist = math.sqrt(dx * dx + dy * dy)

    for radius, score in RINGS:
        if dist <= radius:
            return score

    return 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    # 必要なら左右反転
    # frame = cv2.flip(frame, 1)

    # 常時レーザー位置を検出
    live_point = detect_red_laser(frame)

    # 的中心表示
    cv2.circle(frame, TARGET_CENTER, 5, (255, 0, 0), -1)

    # 得点リング表示
    for radius, score in RINGS:
        cv2.circle(frame, TARGET_CENTER, radius, (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(score),
            (TARGET_CENTER[0] + radius - 20, TARGET_CENTER[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    # 現在のレーザー位置表示
    if live_point is not None:
        cv2.circle(frame, live_point, 10, (0, 255, 255), 2)
        cv2.putText(
            frame,
            "LIVE",
            (live_point[0] + 12, live_point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    # 前回命中位置表示
    if last_hit_point is not None:
        cv2.circle(frame, last_hit_point, 20, (0, 255, 0), 3)

    # シリアル受信
    if ser.in_waiting > 0:
        line = ser.readline().decode(errors="ignore").strip()

        if line == "SHOT":
            shot_count += 1
            point = live_point   # 押した瞬間の現在位置で採点

            if point is not None:
                score = calc_score(point, TARGET_CENTER)
                last_score = score
                total_score += score
                last_hit_point = point
                set_status(f"HIT! {score} pts", (0, 255, 0), 1.2)
            else:
                last_score = 0
                last_hit_point = None
                set_status("NO HIT", (0, 0, 255), 1.2)

        elif line == "RESET":
            last_score = None
            total_score = 0
            shot_count = 0
            last_hit_point = None
            set_status("RESET!", (255, 255, 0), 1.0)

    # 状態表示
    if time.time() < status_until:
        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2
        )

    # スコア表示
    cv2.putText(
        frame,
        f"Last: {last_score if last_score is not None else '-'}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total: {total_score}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Shots: {shot_count}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Laser Shooting Game", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESCで終了
        break

cap.release()
ser.close()
cv2.destroyAllWindows()