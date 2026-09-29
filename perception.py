import time
import cv2
import numpy as np


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    framerate=50,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True max-buffers=1 sync=False"
    )


class Perception:

    def __init__(
        self,
        calib_file="stereo_params.npz",
        bev_scale=2000.0,
        ground_threshold=1.3,
    ):
        try:
            calib = np.load(calib_file)
            self.map1_l = calib["map1_l"]
            self.map2_l = calib["map2_l"]
            self.map1_r = calib["map1_r"]
            self.map2_r = calib["map2_r"]
            print("Perception: キャリブレーションファイルの読み込みに成功しました。")
        except Exception as e:
            print(f"Perception 警告: 補正なしで実行します -> {e}")
            self.map1_l = self.map2_l = self.map1_r = self.map2_r = None

        self.bev_scale = bev_scale
        self.ground_threshold = ground_threshold

        # 処理解像度を落としたため、視差数もそれに合わせて調整
        self.num_disparities = 16 * 4  # 64
        block_size = 7

        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=16,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def process(self, frame_l, frame_r):
        # 1. 元の解像度のまま正確にキャリブレーション（歪み補正・立体視補正）を適用
        if self.map1_l is not None:
            rect_l_orig = cv2.remap(
                frame_l, self.map1_l, self.map2_l, cv2.INTER_LINEAR
            )
            rect_r_orig = cv2.remap(
                frame_r, self.map1_r, self.map2_r, cv2.INTER_LINEAR
            )
        else:
            rect_l_orig, rect_r_orig = frame_l, frame_r

        # 2. 処理負荷を下げるために、ここで初めて画像を1/2に縮小する（例: 1280x720 -> 640x360）
        scale_ratio = 0.5
        rect_l = cv2.resize(
            rect_l_orig, (0, 0), fx=scale_ratio, fy=scale_ratio
        )
        rect_r = cv2.resize(
            rect_r_orig, (0, 0), fx=scale_ratio, fy=scale_ratio
        )

        h, w, _ = rect_l.shape
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        # 縮小画像でステレオマッチング計算（爆速化）
        disparity = (
            self.stereo_matcher.compute(gray_l, gray_r).astype(np.float32)
            / 16.0
        )

        crop_x = self.num_disparities
        disparity_crop = disparity[:, crop_x : w - crop_x]
        rect_l_crop = rect_l[:, crop_x : w - crop_x]
        h_c, w_c = disparity_crop.shape

        disp_clipped = np.clip(disparity_crop, 0, self.num_disparities)
        disp_norm = (disp_clipped / self.num_disparities * 255.0).astype(
            np.uint8
        )
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # 床・障害物判定
        v0 = h_c * 0.20
        v_grid = np.arange(h_c, dtype=np.float32).reshape(-1, 1)
        v_grid = np.repeat(v_grid, w_c, axis=1)

        valid_disp = np.maximum(disparity_crop, 1.0)
        height_ratio = (v_grid - v0) / valid_disp

        diff = height_ratio - self.ground_threshold
        raw_score = 1.0 / (1.0 + np.exp(-diff * 2.5))
        raw_score[v_grid <= v0] = 0.0
        clean_score = (raw_score > 0.40).astype(np.float32)

        # 表示用に元のサイズに戻したオーバーレイを作成
        overlay = rect_l_crop.copy()
        mask_bool = disparity_crop > 0.5
        overlay[mask_bool] = cv2.addWeighted(
            rect_l_crop[mask_bool],
            0.5,
            disp_color[mask_bool],
            0.5,
            0,
        )
        full_overlay = cv2.resize(
            overlay,
            (rect_l_orig.shape[1], rect_l_orig.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # 鳥瞰図（BEV）生成
        bev_w, bev_h = 300, 500
        bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

        u_coords, v_coords = np.meshgrid(np.arange(w_c), np.arange(h_c))
        u_val = u_coords.flatten()
        v_val = v_coords.flatten()
        d_val = disparity_crop.flatten()
        s_val = clean_score.flatten()

        u0 = w_c / 2.0
        SCALE_K = self.bev_scale

        is_ground = (v_val > (v0 + 5)) & (s_val > 0.5)
        d_ground = (v_val[is_ground] - v0) / self.ground_threshold
        z_ground = SCALE_K / np.maximum(d_ground, 0.5)
        x_ground = (u_val[is_ground] - u0) * (z_ground / 150.0)

        base_obs = (s_val <= 0.5) & (d_val >= 1.0) & (d_val < 40.0)
        z_obs = SCALE_K / np.maximum(d_val[base_obs], 0.5)
        x_obs = (u_val[base_obs] - u0) * (z_obs / 150.0)

        x_all = np.concatenate([x_ground, x_obs])
        z_all = np.concatenate([z_ground, z_obs])

        bev_x = (bev_w // 2 + x_all).astype(np.int32)
        bev_z = (bev_h - 10 - z_all).astype(np.int32)

        in_bounds = (
            (bev_x >= 0)
            & (bev_x < bev_w)
            & (bev_z >= 0)
            & (bev_z < bev_h)
        )
        bev_x = bev_x[in_bounds]
        bev_z = bev_z[in_bounds]

        for x, z in zip(bev_x[::2], bev_z[::2]):
            cv2.circle(bev_map, (x, z), 2, (0, 255, 0), -1)

        bot_pos = (bev_w // 2, bev_h - 10)
        pts = np.array(
            [
                [bot_pos[0], bot_pos[1] - 10],
                [bot_pos[0] - 8, bot_pos[1] + 8],
                [bot_pos[0] + 8, bot_pos[1] + 8],
            ],
            np.int32,
        )
        cv2.drawContours(bev_map, [pts], 0, (255, 200, 0), -1)

        perception_data = {
            "rect_l": rect_l_orig,  # 表示は元の高解像度を維持
            "rect_r": rect_r_orig,
            "disp_color": disp_color,
            "full_overlay": full_overlay,
            "bev_map": bev_map,
        }
        return perception_data


if __name__ == "__main__":
    # キャプチャは元の1280x720、50fpsで取得
    cap_l = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    perc = Perception(bev_scale=1500.0, ground_threshold=1.3)
    print("高精度・高速化デバッグを開始します")

    panel_w, panel_h = 400, 225
    bev_w = 300
    total_h = panel_h * 2

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        data = perc.process(frame_l, frame_r)

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            current_fps = 1.0 / dt
            fps_smooth = fps_smooth * 0.8 + current_fps * 0.2

        img_left = cv2.resize(data["rect_l"], (panel_w, panel_h))
        img_right = cv2.resize(data["rect_r"], (panel_w, panel_h))
        img_disp = cv2.resize(data["disp_color"], (panel_w, panel_h))
        img_result = cv2.resize(data["full_overlay"], (panel_w, panel_h))

        img_bev = cv2.resize(data["bev_map"], (bev_w, total_h))
        cv2.putText(
            img_bev,
            f"FPS: {fps_smooth:.1f} Hz",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        top_row = np.hstack((img_left, img_right))
        bottom_row = np.hstack((img_disp, img_result))
        left_grid = np.vstack((top_row, bottom_row))
        full_window = np.hstack((left_grid, img_bev))

        cv2.imshow("Perception Multi-Debug", full_window)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()