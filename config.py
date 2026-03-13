CONFIG = {

    # ===== 画面 =====
    "screen_w": 1280,
    "screen_h": 720,
    "fullscreen": False,
    "window_title": "Laser Balloon Game",

    # ===== ゲーム基本 =====
    "game_time_sec": 500,
    "spawn_interval": 0.8,
    "max_balloons": 12,

    # ===== 風船サイズ =====
    "balloon_radius_min": 25,
    "balloon_radius_max": 50,

    # ===== 風船速度 =====
    "balloon_speed_min": 100,
    "balloon_speed_max": 300,

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
    "score_bomb": -10,

    # ===== コンボ =====
    "combo_x2_count": 5,
    "combo_x3_count": 10,
    "combo_x2_multiplier": 2,
    "combo_x3_multiplier": 3,

    # ===== レーザー検出 =====
    "laser_threshold": 205,
    "laser_min_area": 0,
    "laser_max_area": 2,
    "laser_v_min": 180,
    "laser_r_min": 60,
    "laser_red_margin": 28,
    "laser_response_min": 10,
    "laser_brightness_gain": 1.15,
    "laser_red_response_gain": 1.0,
    "laser_local_bg_kernel": 7,
    "laser_peak_window_radius": 4,
    "laser_peak_relative_threshold": 0.45,
    "laser_screen_smoothing": 0.35,
    "laser_raw_response_gain": 0.75,
    "laser_column_bias_gain": 0.8,
    "laser_column_bias_kernel": 81,
    "laser_temporal_gain": 0.65,
    "laser_bg_alpha": 0.04,
    "laser_tracking_search_radius": 110,
    "laser_tracking_min_score_ratio": 0.45,
    "laser_tracking_relaxed_min": 3,
    "laser_ignore_left_px": 0,
    "laser_ignore_right_px": 0,
    "laser_ignore_top_px": 0,
    "laser_ignore_bottom_px": 0,
    "laser_screen_offset_x": 0,
    "laser_screen_offset_y": 18,
    "laser_hold_sec": 0.12,

    # ===== カメラ =====
    "camera_index": 1,
    "camera_backend": "dshow",
    "camera_probe_indices": [0, 1, 2, 3, 4, 5],
    "camera_buffer_size": 1,
    "camera_w": 1280,
    "camera_h": 720,
    "camera_auto_exposure": 0.25,
    "camera_exposure": -7,
    "camera_gain": 0,
    "camera_brightness": 20,
    "shot_frame_grabs": 2,
    "show_camera_window": False,
    "show_threshold_window": False,
    "show_tracking_window": True,
    "show_calibration_overlay": True,
    "tracking_window_scale": 0.6,

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
    "shot_hit_color": (90, 255, 180),
    "shot_miss_color": (80, 200, 255),
    "enable_sound": True,
    "sound_volume": 0.8,

    # ===== カーソル =====
    "show_projected_laser_cursor": False,
    "laser_cursor_color": (90, 220, 255),

    # ===== 風船色 =====
    "balloon_normal_color": (18, 70, 105),
    "balloon_bonus_color": (28, 110, 120),
    "balloon_bomb_color": (32, 42, 58),
    "balloon_string_color": (45, 58, 74),
    "balloon_normal_outline_color": (90, 175, 215),
    "balloon_bonus_outline_color": (120, 230, 240),
    "balloon_bomb_outline_color": (125, 140, 165),
    "balloon_outline_width": 1,

    # ===== 破裂エフェクト =====
    "pop_effect_duration_sec": 0.28,
    "pop_particle_count": 12,
    "pop_particle_speed_min": 140,
    "pop_particle_speed_max": 280,
    "pop_particle_gravity": 420,
    "pop_flash_color": (160, 220, 255),

    # ===== UI =====
    "bg_color": (7, 10, 18),
    "text_color": (165, 190, 215),
    "show_ui_text": False,
    "projector_dim_alpha": 120,

    # ===== 投影補正 =====
    # カメラ画像上のスクリーン4点
    # 左上, 右上, 右下, 左下
    "cam_points_ref_w": 1280,
    "cam_points_ref_h": 720,
    "cam_point_step": 2,
    "cam_point_step_fast": 10,
    "auto_calibration_canny_low": 40,
    "auto_calibration_canny_high": 120,
    "auto_calibration_min_area_ratio": 0.12,
    "auto_calibration_epsilon_ratio": 0.03,
    "cam_points": [
        [202, 82],
        [1060, 110],
        [1040, 606],
        [202, 600],
    ],

}
