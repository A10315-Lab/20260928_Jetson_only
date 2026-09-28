import cv2
import numpy as np


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    framerate=30,
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
            print("Perception: stereo_params.npz の読み込みに成功しました。")
        except Exception as e:
            print(
                f"Perception 警告: キャリブレーションファイルなし。補正なしで実行します -> {e}"
            )
            self.map1_l = self.map2_l = self.map1_r = self.map2_r = None

        self.bev_scale = bev_scale
        self.ground_threshold = ground_threshold

        self.num_disparities = 16 * 8
        block_size = 11

        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=2,
            uniquenessRatio=5,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def process(self, frame_l, frame_r):
        if self.map1_l is not None:
            rect_l = cv2.remap(
                frame_l, self.map1_l, self.map2_l, cv2.INTER_LINEAR
            )
            rect_r = cv2.remap(
                frame_r, self.map1_r, self.map2_r, cv2.INTER_LINEAR
            )
        else:
            rect_l, rect_r = frame_l, frame_r

        h, w, _ = rect_l.shape
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        disparity = (
            self.stereo_matcher.compute(gray_l, gray_r).astype(np.float32)
            / 16.0
        )

        # ★ 左右均等に num_disparities 分クロップして左端の帯ノイズを排除 ★
        crop_x = self.num_disparities
        crop_w = w - (crop_x * 2)

        disparity_crop = disparity[:, crop_x : w - crop_x]
        rect_l_crop = rect_l[:, crop_x : w - crop_x]

        disp_clipped = np.clip(disparity_crop, 0, self.num_disparities)
        disp_norm = (disp_clipped / self.num_disparities * 255.0).astype(
            np.uint8
        )
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        proc_w = 320
        scale = proc_w / crop_w
        proc_h = int(h * scale)

        small_roi = cv2.resize(
            rect_l_crop, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST
        )
        small_disp = cv2.resize(
            disparity_crop,
            (proc_w, proc_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # 1. 床/障害物判定
        disp_smooth = cv2.GaussianBlur(small_disp, (9, 9), 0)
        v_grid = np.arange(proc_h, dtype=np.float32).reshape(-1, 1)
        v_grid = np.repeat(v_grid, proc_w, axis=1)

        v0 = proc_h * 0.20  # 消失点Y
        valid_disp = np.maximum(disp_smooth, 1.0)
        height_ratio = (v_grid - v0) / valid_disp

        diff = height_ratio - self.ground_threshold
        raw_score = 1.0 / (1.0 + np.exp(-diff * 2.5))

        # 消失点より上は床ではない
        raw_score[v_grid <= v0] = 0.0

        score_u8 = (raw_score * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        score_u8 = cv2.morphologyEx(score_u8, cv2.MORPH_OPEN, kernel)
        score_u8 = cv2.morphologyEx(score_u8, cv2.MORPH_CLOSE, kernel)
        clean_score = score_u8.astype(np.float32) / 255.0

        # オーバーレイ作成
        hue = (clean_score * 60.0).astype(np.uint8)
        sat = np.full_like(hue, 255)
        val = np.full_like(hue, 255)
        hsv_map = cv2.merge([hue, sat, val])
        grad_bgr = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)

        valid_mask = (small_disp > 0.5).astype(np.uint8) * 255
        mask_bool = valid_mask > 0

        overlay = small_roi.copy()
        overlay[mask_bool] = cv2.addWeighted(
            small_roi[mask_bool], 0.35, grad_bgr[mask_bool], 0.65, 0
        )
        full_overlay = cv2.resize(
            overlay, (w, h), interpolation=cv2.INTER_NEAREST
        )

        # 2. 鳥瞰図（BEV）生成
        bev_w, bev_h = 360, 640
        bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

        u_coords, v_coords = np.meshgrid(
            np.arange(proc_w), np.arange(proc_h)
        )

        u_val = u_coords.flatten()
        v_val = v_coords.flatten()
        d_val = small_disp.flatten()
        s_val = clean_score.flatten()

        u0 = proc_w / 2.0
        SCALE_K = self.bev_scale

        # --- A. 床 (緑領域) のプロット ---
        is_ground = (v_val > (v0 + 5)) & (s_val > 0.40)
        v_diff_g = v_val[is_ground] - v0
        d_ground = v_diff_g / self.ground_threshold
        z_ground = SCALE_K / np.maximum(d_ground, 0.5)
        x_ground = (u_val[is_ground] - u0) * (z_ground / 180.0)
        s_ground = s_val[is_ground]

        # --- B. 障害物/壁 (赤領域) のプロット ---
        u_center_min = proc_w * 0.30
        u_center_max = proc_w * 0.70
        is_center = (u_val >= u_center_min) & (u_val <= u_center_max)

        base_obs = (s_val <= 0.40) & (d_val >= 1.2) & (d_val < 60.0)

        center_obs = base_obs & is_center & (v_val > (v0 + 10))
        side_obs = base_obs & (~is_center)

        is_obstacle = center_obs | side_obs

        z_obs = SCALE_K / np.maximum(d_val[is_obstacle], 0.5)
        x_obs = (u_val[is_obstacle] - u0) * (z_obs / 180.0)
        s_obs = s_val[is_obstacle]

        x_all = np.concatenate([x_ground, x_obs])
        z_all = np.concatenate([z_ground, z_obs])
        s_all = np.concatenate([s_ground, s_obs])

        bev_x = (bev_w // 2 + x_all).astype(np.int32)
        bev_z = (bev_h - 20 - z_all).astype(np.int32)

        bev_z = np.clip(bev_z, 5, bev_h - 1)

        in_bounds = (bev_x >= 0) & (bev_x < bev_w)

        bev_x = bev_x[in_bounds]
        bev_z = bev_z[in_bounds]
        s_all = s_all[in_bounds]

        sort_idx = np.argsort(s_all)[::-1]

        for x, z, score in zip(
            bev_x[sort_idx], bev_z[sort_idx], s_all[sort_idx]
        ):
            color = (0, int(score * 255), int((1.0 - score) * 255))
            cv2.circle(bev_map, (x, z), 2, color, -1)

        # 自機マーク（青三角形）
        bot_pos = (bev_w // 2, bev_h - 15)
        pts = np.array(
            [
                [bot_pos[0], bot_pos[1] - 15],
                [bot_pos[0] - 12, bot_pos[1] + 10],
                [bot_pos[0] + 12, bot_pos[1] + 10],
            ],
            np.int32,
        )
        cv2.drawContours(bev_map, [pts], 0, (255, 200, 0), -1)

        # 距離グリッド線
        for r in range(60, bev_h - 20, 80):
            cv2.line(
                bev_map,
                (0, bev_h - 20 - r),
                (bev_w, bev_h - 20 - r),
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )

        # 壁・偏り判定
        h_d, w_d = disparity_crop.shape
        left_area = disparity_crop[:, : int(w_d * 0.35)]
        right_area = disparity_crop[:, int(w_d * 0.65) :]

        left_cont = np.sum((left_area > 3.0) & (left_area < 60.0)) / (
            h_d * w_d * 0.35 + 1e-5
        )
        right_cont = np.sum((right_area > 3.0) & (right_area < 60.0)) / (
            h_d * w_d * 0.35 + 1e-5
        )
        stereo_bias = float(left_cont - right_cont) * 50.0

        if left_cont > right_cont:
            wall_yaw_error = float(
                np.nanmean(left_area[: int(h_d * 0.5), :])
                - np.nanmean(left_area[int(h_d * 0.5) :, :])
            )
        else:
            wall_yaw_error = float(
                np.nanmean(right_area[int(h_d * 0.5) :, :])
                - np.nanmean(right_area[: int(h_d * 0.5), :])
            )
        if np.isnan(wall_yaw_error):
            wall_yaw_error = 0.0

        perception_data = {
            "rect_l": rect_l,
            "rect_r": rect_r,
            "disp_color": disp_color,
            "stereo_bias": stereo_bias,
            "wall_yaw_error": wall_yaw_error,
            "full_overlay": full_overlay,
            "bev_map": bev_map,
            "proc_h": proc_h,
            "proc_w": proc_w,
        }
        return perception_data


if __name__ == "__main__":
    cap_l = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    perc = Perception(bev_scale=2000.0, ground_threshold=1.3)
    print("デバッグを開始します")

    panel_w, panel_h = 480, 270
    bev_w = 380
    total_h = panel_h * 2

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        data = perc.process(frame_l, frame_r)

        img_left = cv2.resize(data["rect_l"], (panel_w, panel_h))
        cv2.putText(
            img_left,
            "Left Camera",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        img_right = cv2.resize(data["rect_r"], (panel_w, panel_h))
        cv2.putText(
            img_right,
            "Right Camera",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        img_disp = cv2.resize(data["disp_color"], (panel_w, panel_h))
        cv2.putText(
            img_disp,
            "Disparity Map (Depth)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        result_img = data["full_overlay"]
        img_result = cv2.resize(result_img, (panel_w, panel_h))
        cv2.putText(
            img_result,
            f"G-Thresh: {perc.ground_threshold:.2f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        img_bev = cv2.resize(data["bev_map"], (bev_w, total_h))
        cv2.putText(
            img_bev,
            f"BEV (Scale: {perc.bev_scale:.0f})",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
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
        elif key in [ord("+"), ord("=")]:
            perc.bev_scale = min(perc.bev_scale + 200.0, 5000.0)
        elif key in [ord("-"), ord("_")]:
            perc.bev_scale = max(perc.bev_scale - 200.0, 500.0)
        elif key == ord("["):
            perc.ground_threshold = max(perc.ground_threshold - 0.1, 0.5)
        elif key == ord("]"):
            perc.ground_threshold = min(perc.ground_threshold + 0.1, 4.0)

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()