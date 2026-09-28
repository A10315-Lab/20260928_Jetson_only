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
    def __init__(self, calib_file="stereo_params.npz"):
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)
        # 垂直判定用（少し太くしてノイズを弾く）
        self.kernel_vertical = np.ones((12, 2), np.uint8)

        # --- 時系列フィルタ用パラメータ ---
        self.prev_hsv = None
        self.alpha = 0.15

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

        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 5,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
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

        # 認識領域（上部20%〜85%）
        roi_top, roi_bottom = int(h * 0.20), int(h * 0.85)
        roi = rect_l[roi_top:roi_bottom, :]
        proc_w = 320
        scale = proc_w / w
        proc_h = int(roi.shape[0] * scale)

        small_roi = cv2.resize(
            roi, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST
        )
        small_disp = cv2.resize(
            disparity[roi_top:roi_bottom, :],
            (proc_w, proc_h),
            interpolation=cv2.INTER_LINEAR,
        )
        hsv_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)

        # ==========================================================
        # ★ 1. 安全な「超足元中央」からのステレオシード抽出 ★
        # ==========================================================
        # 人の脚などが入りにくい「真下の中央20%」だけを厳選
        sample_y_start = int(proc_h * 0.82)
        sample_x_start = int(proc_w * 0.40)
        sample_x_end = int(proc_w * 0.60)

        stereo_seed = np.zeros((proc_h, proc_w), dtype=np.uint8)

        base_patch = small_disp[
            sample_y_start:proc_h, sample_x_start:sample_x_end
        ]
        valid_disp = base_patch[base_patch > 0]
        base_median_disp = (
            np.median(valid_disp) if len(valid_disp) > 5 else 15.0
        )

        for y in range(sample_y_start, proc_h):
            factor = (proc_h - y) / float(proc_h - sample_y_start + 1e-5)
            min_d = max(1.0, base_median_disp * (factor * 0.4 + 0.6))
            max_d = base_median_disp * 1.3 + 4.0
            row_disp = small_disp[y, :]
            stereo_seed[y, (row_disp >= min_d) & (row_disp <= max_d)] = 255

        cv2.morphologyEx(
            stereo_seed, cv2.MORPH_OPEN, self.kernel_open, dst=stereo_seed
        )

        # ==========================================================
        # ★ 2. カラーモデル学習（許容幅の厳格な上限キャップ付き） ★
        # ==========================================================
        seed_pixels = hsv_roi[stereo_seed > 0]

        if len(seed_pixels) > 30:
            mean_hsv = np.mean(seed_pixels, axis=0)
            std_hsv = np.std(seed_pixels, axis=0)

            # 許容幅が広がりすぎないように「絶対上限（min/max）」をかける
            tol_h = min(12, max(6, int(std_hsv[0] * 1.5)))
            tol_s = min(30, max(15, int(std_hsv[1] * 1.8)))
            tol_v = min(30, max(20, int(std_hsv[2] * 1.8)))

            curr_hsv = mean_hsv
        else:
            curr_hsv = (
                self.prev_hsv
                if self.prev_hsv is not None
                else np.array([0, 0, 100], dtype=np.float32)
            )
            tol_h, tol_s, tol_v = 10, 25, 25

        if self.prev_hsv is None:
            self.prev_hsv = curr_hsv
        else:
            self.prev_hsv = (
                self.alpha * curr_hsv + (1.0 - self.alpha) * self.prev_hsv
            )

        mean_h, mean_s, mean_v = self.prev_hsv

        lower_bound = np.array(
            [
                max(0, int(mean_h - tol_h)),
                max(0, int(mean_s - tol_s)),
                max(0, int(mean_v - tol_v)),
            ],
            dtype=np.uint8,
        )

        upper_bound = np.array(
            [
                min(180, int(mean_h + tol_h)),
                min(255, int(mean_s + tol_s)),
                min(255, int(mean_v + tol_v)),
            ],
            dtype=np.uint8,
        )

        mask_color = cv2.inRange(hsv_roi, lower_bound, upper_bound)

        # ==========================================================
        # ★ 3. 連結成分抽出（暴走ガード付き） ★
        # ==========================================================
        # ステレオシードがない場合は強引に広げない
        if np.sum(stereo_seed) > 0:
            mask_combined = cv2.bitwise_or(mask_color, stereo_seed)
            cv2.morphologyEx(
                mask_combined,
                cv2.MORPH_CLOSE,
                self.kernel_close,
                dst=mask_combined,
            )

            num_labels, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(mask_combined)
            )
            mask_ground = np.zeros((proc_h, proc_w), dtype=np.uint8)

            seed_labels = np.unique(labels[stereo_seed > 0])
            for label in seed_labels:
                if label == 0:
                    continue
                mask_ground[labels == label] = 255
        else:
            mask_ground = stereo_seed.copy()

        # ==========================================================
        # ★ 4. 垂直連続視差（車・壁など）のノイズ抑制付き検出 ★
        # ==========================================================
        # メディアンフィルタでノイズ成分を強力カット
        disp_clean = cv2.medianBlur(small_disp.astype(np.float32), 5)
        sobely_disp = cv2.Sobel(disp_clean, cv2.CV_32F, 0, 1, ksize=3)

        # 一定以上の強い視差（＝近い物体）かつ垂直面である場合のみ
        vertical_surface = (
            (disp_clean > 10.0) & (np.abs(sobely_disp) < 0.25)
        ).astype(np.uint8) * 255

        # 縦長カーネルでノイズを排除
        wall_disp_mask = cv2.morphologyEx(
            vertical_surface, cv2.MORPH_OPEN, self.kernel_vertical
        )

        # Cannyエッジの閾値を上げてノイズでの誤反応を防止 (120, 220)
        edges = cv2.Canny(cv2.GaussianBlur(small_roi, (5, 5), 0), 120, 220)
        edges[int(proc_h * 0.80) :, :] = 0  # 足元は除外

        # 壁マスク合成
        wall_mask = cv2.bitwise_or(
            wall_disp_mask, cv2.dilate(edges, np.ones((3, 3), np.uint8))
        )

        # 床優先で壁から削る
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(mask_ground))

        # ==========================================================
        # ★ 5. 壁の評価（バイアス・ヨー角） ★
        # ==========================================================
        disp_roi = disparity[roi_top:roi_bottom, :]
        h_d, w_d = disp_roi.shape
        left_area = disp_roi[:, : int(w_d * 0.35)]
        right_area = disp_roi[:, int(w_d * 0.65) :]

        left_cont = np.sum((left_area > 5.0) & (left_area < 60.0)) / (
            h_d * w_d * 0.35 + 1e-5
        )
        right_cont = np.sum((right_area > 5.0) & (right_area < 60.0)) / (
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

        # デバッグ描画
        overlay = small_roi.copy()
        # 床（緑）
        overlay[mask_ground > 0] = (
            overlay[mask_ground > 0] * 0.5
            + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        )
        # 車・壁（赤）
        overlay[wall_mask > 0] = (
            overlay[wall_mask > 0] * 0.5
            + np.array([0, 0, 255], dtype=np.uint8) * 0.5
        )
        # 確定シード（シアン）
        overlay[stereo_seed > 0] = (
            overlay[stereo_seed > 0] * 0.2
            + np.array([255, 255, 0], dtype=np.uint8) * 0.8
        )

        full_overlay = cv2.resize(
            overlay, (w, roi.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        perception_data = {
            "rect_l": rect_l,
            "mask_ground": mask_ground,
            "wall_mask": wall_mask,
            "scale": scale,
            "roi_top": roi_top,
            "roi_bottom": roi_bottom,
            "stereo_bias": stereo_bias,
            "wall_yaw_error": wall_yaw_error,
            "full_overlay": full_overlay,
            "proc_h": proc_h,
            "proc_w": proc_w,
        }
        return perception_data


if __name__ == "__main__":
    cap_l = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)
    perc = Perception()
    print("認識部のデバッグを開始します（'q'で終了）")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        data = perc.process(frame_l, frame_r)

        disp = data["rect_l"].copy()
        alpha = 0.4
        disp[data["roi_top"] : data["roi_bottom"], :] = cv2.addWeighted(
            disp[data["roi_top"] : data["roi_bottom"], :],
            1 - alpha,
            data["full_overlay"],
            alpha,
            0,
        )

        cv2.putText(
            disp,
            f"Bias: {data['stereo_bias']:.2f} YawErr: {data['wall_yaw_error']:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Perception Debug", cv2.resize(disp, (640, 360)))

        if cv2.waitKey(1) == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()