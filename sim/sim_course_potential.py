"""
TT-02 2D Course Simulator with Dynamic Risk Potential Field Controller
- コース幅: 1.0 m
- 車体: Tamiya TT-02 相当 (Wheelbase = 0.257 m, Width = 0.185 m)
- コントローラ: ポンサトーン型リスクポテンシャル法
"""
import numpy as np
import cv2
import math
from scipy.interpolate import CubicSpline


class PotentialControllerForSim:
    def __init__(self, k_lane=0.8, k_curv=0.4, k_wall=1.8, k_yaw=0.5, b_safe=0.25, sigma_w=0.15):
        self.k_lane = k_lane    # 走路中心追従引力ゲイン
        self.k_curv = k_curv    # カーブ予測ゲイン
        self.k_wall = k_wall    # 壁面斥力ゲイン
        self.k_yaw = k_yaw      # ヨーダンピングゲイン
        self.b_safe = b_safe    # 斥力安全マージン (0.0=中央, 1.0=壁接触)
        self.sigma_w = sigma_w

    def compute_steer_angle(self, offset, curvature, stereo_bias, wall_yaw_error):
        # 1. 引力ポテンシャル
        preview_offset = offset + (self.k_curv * curvature)
        f_att = - self.k_lane * preview_offset

        # 2. 斥力ポテンシャル (指数関数型)
        f_rep = 0.0
        abs_bias = abs(stereo_bias)
        if abs_bias > self.b_safe:
            exponent = min((abs_bias - self.b_safe) / self.sigma_w, 4.0)
            rep_mag = self.k_wall * (math.exp(exponent) - 1.0)
            # stereo_bias > 0 (右壁が近い) -> 左旋回 (delta > 0)
            sign = 1.0 if stereo_bias > 0 else -1.0
            f_rep = sign * rep_mag

        # 3. ヨー角ダンピング
        f_yaw = - self.k_yaw * wall_yaw_error

        # TT-02 の最大舵角 (約 28度 = 0.488 rad) で飽和
        steer_rad = f_att + f_rep + f_yaw
        steer_rad = np.clip(steer_rad, -np.radians(28), np.radians(28))
        return steer_rad, f_att, f_rep, f_yaw


class TT02Vehicle:
    """Tamiya TT-02 2輪キネマティック・バイシクルモデル"""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, speed=1.2):
        self.x = x            # [m]
        self.y = y            # [m]
        self.yaw = yaw        # [rad]
        self.v = speed        # [m/s] 一定速度 (約 4.3 km/h)
        self.L = 0.257        # ホイールベース [m]
        self.width = 0.185    # トレッド/車幅 [m]
        self.steer = 0.0      # 現在舵角 [rad]

    def update(self, steer_cmd, dt=0.033):
        steering_speed = np.radians(300)  # サーボ速度 (rad/s)
        steer_diff = steer_cmd - self.steer
        max_step = steering_speed * dt
        self.steer += np.clip(steer_diff, -max_step, max_step)

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * math.tan(self.steer) * dt
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi


def generate_random_course(num_points=10, scale=9.0, track_width=1.0):
    """ランダムな閉曲線の滑らかなコースを生成"""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = scale + np.random.uniform(-scale * 0.3, scale * 0.3, size=num_points)
    
    ctrl_x = radii * np.cos(angles)
    ctrl_y = radii * np.sin(angles)

    # 周期スプライン用に始点を末尾に追加（最初と最後の点を完全一致させる）
    ctrl_x = np.append(ctrl_x, ctrl_x[0])
    ctrl_y = np.append(ctrl_y, ctrl_y[0])

    t = np.linspace(0, 1, num_points + 1)
    t_fine = np.linspace(0, 1, 800, endpoint=False)

    cs_x = CubicSpline(t, ctrl_x, bc_type='periodic')
    cs_y = CubicSpline(t, ctrl_y, bc_type='periodic')

    cx = cs_x(t_fine)
    cy = cs_y(t_fine)
    center_line = np.vstack([cx, cy]).T

    # 接線ベクトルと法線ベクトルから左右の壁（コース幅 1.0m）を計算
    tangents = np.gradient(center_line, axis=0)
    normals = np.vstack([-tangents[:, 1], tangents[:, 0]]).T
    norms = np.linalg.norm(normals, axis=1)[:, np.newaxis]
    normals /= np.where(norms == 0, 1.0, norms)

    half_w = track_width / 2.0
    left_wall = center_line + normals * half_w
    right_wall = center_line - normals * half_w

    return center_line, left_wall, right_wall


def run_simulation():
    track_width = 1.0
    center_line, left_wall, right_wall = generate_random_course(num_points=10, scale=9.0, track_width=track_width)
    
    start_yaw = math.atan2(center_line[1, 1] - center_line[0, 1], center_line[1, 0] - center_line[0, 0])
    car = TT02Vehicle(x=center_line[0, 0], y=center_line[0, 1], yaw=start_yaw, speed=1.2)
    controller = PotentialControllerForSim()

    IMG_SIZE = 800
    PPM = 28  # Pixels Per Meter
    OFFSET = IMG_SIZE // 2

    def to_img(x, y):
        u = int(OFFSET + x * PPM)
        v = int(OFFSET - y * PPM)
        return (u, v)

    trajectory = []
    print("=== TT-02 走行シミュレーション開始 ('q'キーで終了, 'r'キーでコース再生成) ===")

    dt = 0.033
    while True:
        # --- A. センサ値の擬似計測 ---
        dists = np.linalg.norm(center_line - np.array([car.x, car.y]), axis=1)
        idx = np.argmin(dists)
        
        next_idx = (idx + 6) % len(center_line)
        tangent_angle = math.atan2(center_line[next_idx, 1] - center_line[idx, 1], 
                                   center_line[next_idx, 0] - center_line[idx, 0])
        yaw_err = (car.yaw - tangent_angle + np.pi) % (2 * np.pi) - np.pi

        dx = car.x - center_line[idx, 0]
        dy = car.y - center_line[idx, 1]
        lat_offset = -dx * math.sin(tangent_angle) + dy * math.cos(tangent_angle)

        idx_prev = (idx - 6) % len(center_line)
        v1 = center_line[idx] - center_line[idx_prev]
        v2 = center_line[next_idx] - center_line[idx]
        angle_diff = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        curvature = angle_diff / max(np.linalg.norm(v1), 1e-4)

        d_left = (track_width / 2.0) + lat_offset
        d_right = (track_width / 2.0) - lat_offset
        stereo_bias = (d_left - d_right) / track_width

        # --- B. 制御計算 ---
        steer_cmd, f_att, f_rep, f_yaw = controller.compute_steer_angle(
            offset=lat_offset, 
            curvature=curvature, 
            stereo_bias=stereo_bias, 
            wall_yaw_error=yaw_err
        )

        # --- C. 車両状態の更新 ---
        car.update(steer_cmd, dt)
        trajectory.append((car.x, car.y))
        if len(trajectory) > 250:
            trajectory.pop(0)

        collision = abs(lat_offset) > (track_width / 2.0 - car.width / 2.0)

        # --- D. 描画 ---
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        pts_l = np.array([to_img(p[0], p[1]) for p in left_wall], dtype=np.int32)
        pts_r = np.array([to_img(p[0], p[1]) for p in right_wall], dtype=np.int32)
        pts_c = np.array([to_img(p[0], p[1]) for p in center_line], dtype=np.int32)
        cv2.polylines(canvas, [pts_l], isClosed=True, color=(140, 140, 140), thickness=2)
        cv2.polylines(canvas, [pts_r], isClosed=True, color=(140, 140, 140), thickness=2)
        cv2.polylines(canvas, [pts_c], isClosed=True, color=(50, 50, 50), thickness=1)

        if len(trajectory) > 1:
            traj_pts = np.array([to_img(p[0], p[1]) for p in trajectory], dtype=np.int32)
            cv2.polylines(canvas, [traj_pts], isClosed=False, color=(0, 215, 255), thickness=2)

        car_pos = to_img(car.x, car.y)
        car_col = (0, 0, 255) if collision else (255, 120, 0)
        cv2.circle(canvas, car_pos, int(car.width * PPM / 2), car_col, -1)
        
        front_vec = (int(car_pos[0] + math.cos(car.yaw + car.steer) * 20),
                     int(car_pos[1] - math.sin(car.yaw + car.steer) * 20))
        cv2.line(canvas, car_pos, front_vec, (0, 255, 0), 2)

        cv2.putText(canvas, f"Speed: {car.v:.1f} m/s | Track: {track_width:.1f}m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Offset: {lat_offset*100:+.1f} cm", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, f"Bias: {stereo_bias:+.2f} | YawErr: {math.degrees(yaw_err):+.1f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, f"F_att:{f_att:+.2f}  F_rep:{f_rep:+.2f}  F_yaw:{f_yaw:+.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        if collision:
            cv2.putText(canvas, "COLLISION DETECTED!", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("TT-02 Potential Field Simulation", canvas)
        key = cv2.waitKey(int(dt * 1000)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            center_line, left_wall, right_wall = generate_random_course(num_points=10, scale=9.0, track_width=track_width)
            start_yaw = math.atan2(center_line[1, 1] - center_line[0, 1], center_line[1, 0] - center_line[0, 0])
            car = TT02Vehicle(x=center_line[0, 0], y=center_line[0, 1], yaw=start_yaw, speed=1.2)
            trajectory.clear()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_simulation()