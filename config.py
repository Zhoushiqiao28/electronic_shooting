CONFIG = {

    # ===== 画面 =====
    "screen_w": 1280,
    "screen_h": 720,
    "fullscreen": False,
    "window_title": "Laser Balloon Game",

    # ===== ゲーム基本 =====
    "game_time_sec": 1000,
    "spawn_interval": 0.8,
    "max_balloons": 12,

    # ===== 風船サイズ =====
    "balloon_radius_min": 25,
    "balloon_radius_max": 45,

    # ===== 風船速度 =====
    "balloon_speed_min": 120,
    "balloon_speed_max": 220,

    # ===== スポーン位置 =====
    "spawn_x_margin": 80,
    "spawn_y_min": 20,
    "spawn_y_max": 100,

    # ===== 出現率 =====
    "spawn_rate_normal": 0.75,
    "spawn_rate_bonus": 0.18,
    "spawn_rate_bomb": 0.07,

    # ===== 得点 =====
    "score_normal": 10,
    "score_bonus": 30,
    "score_bomb": -20,

    # ===== コンボ =====
    "combo_x2_count": 5,
    "combo_x3_count": 10,
    "combo_x2_multiplier": 2,
    "combo_x3_multiplier": 3,

    # ===== レーザー検出 =====
    "laser_threshold": 205,
    "laser_min_area": 0,
    "laser_max_area": 200,

    # ===== カメラ =====
    "camera_index": 0,
    "show_camera_window": False,
    "show_threshold_window": False,
    "show_tracking_window": True,
    "show_calibration_overlay": True,

    # ===== シリアル =====
    "use_serial": True,
    "serial_port": "COM10",
    "baudrate": 115200,
    "shot_cooldown_sec": 0.15,

    # ===== ショット表示 =====
    "show_shot_marker": True,
    "shot_marker_duration_sec": 0.2,
    "camera_feedback_duration_sec": 0.8,
    "shot_marker_radius": 18,
    "shot_hit_color": (0, 255, 0),
    "shot_miss_color": (255, 0, 0),

    # ===== カーソル =====
    "laser_cursor_color": (255, 255, 255),

    # ===== 風船色 =====
    "balloon_normal_color": (220, 60, 60),
    "balloon_bonus_color": (240, 210, 30),
    "balloon_bomb_color": (60, 60, 60),
    "balloon_string_color": (220, 220, 220),

    # ===== UI =====
    "bg_color": (20, 20, 35),
    "text_color": (255, 255, 255),

    # ===== 投影補正 =====
    # カメラ画像上のスクリーン4点
    # 左上, 右上, 右下, 左下
    "cam_points": [
        [100, 100],
        [1180, 90],
        [1190, 680],
        [90, 690],
    ],

}
