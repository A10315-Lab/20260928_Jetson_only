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

        self.num_disparities = 16 * 5  # 80px (探索幅)
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=7,
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
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

        # 左右均等クロップ処理 (両端 80px カット)
        crop_x = self.num_disparities
        crop_w = w - (crop_x * 2)

        disparity_crop = disparity[:, crop_x : w - crop_x]
        rect_l_crop = rect_l[:, crop_x : w - crop_x]

        # 視差マップのカラー化（デバッグ用）
        disp_valid = np.maximum(0, disparity_crop)
        disp_norm = cv2.normalize(
            disp_valid,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # 処理用リサイズ (320x180相当)
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

        # ==========================================================
        # ★ 3D幾何学（高さ比率）による床/障害物判定 ★
        # ==========================================================
        # 1. 視差マップのノイズ除去
        disp_smooth = cv2.GaussianBlur(small_disp, (9, 9), 0)

        # 2. 画像の各Y座標（v）グリッド生成
        v_grid = np.arange(proc_h, dtype=np.float32).reshape(-1, 1)
        v_grid = np.repeat(v_grid, proc_w, axis=1)

        # 3. 画面の消失点 Y0 (カメラの仰俯角に合わせて微調整可能, 画面上部〜中央)
        v0 = proc_h * 0.35

        # 4. 幾何学的な高さ比率 H_ratio = (v - v0) / max(d, 1.0)
        valid_disp = np.maximum(disp_smooth, 1.0)
        height_ratio = (v_grid - v0) / valid_disp

        # 5. ベッド/床面の閾値判定
        # 平らな床/ベッド面では height_ratio が高くなり、突起物/壁では低くなります
        ground_threshold = 2.2  # ベッド面の高さ比率基準

        # 6. シグモイド関数で「赤（障害物:0.0）」と「緑（床:1.0）」に綺麗に二分化
        diff = height_ratio - ground_threshold
        raw_score = 1.0 / (1.0 + np.exp(-diff * 2.5))

        # 7. モルフォロジー処理で微小なモザイクノイズを除去
        score_u8 = (raw_score * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        score_u8 = cv2.morphologyEx(score_u8, cv2.MORPH_OPEN, kernel)
        score_u8 = cv2.morphologyEx(score_u8, cv2.MORPH_CLOSE, kernel)

        clean_score = score_u8.astype(np.float32) / 255.0

        # ==========================================================
        # ★ HSVグラデーション作成 ★
        # ==========================================================
        # H: 60(緑: ベッド面) -> 0(赤: ダンボール/壁/障害物)
        hue = (clean_score * 60.0).astype(np.uint8)
        sat = np.full_like(hue, 255)
        val = np.full_like(hue, 255)

        hsv_map = cv2.merge([hue, sat, val])
        grad_bgr = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)

        # 有効視差領域の重ね合わせ（視差 2.0 以上のみ表示）
        valid_mask = (small_disp > 2.0).astype(np.uint8) * 255
        mask_bool = valid_mask > 0

        overlay = small_roi.copy()
        overlay[mask_bool] = cv2.addWeighted(
            small_roi[mask_bool], 0.35, grad_bgr[mask_bool], 0.65, 0
        )

        full_overlay = cv2.resize(
            overlay, (w, h), interpolation=cv2.INTER_NEAREST
        )

        # ==========================================================
        # ★ 評価値（左右バイアス） ★
        # ==========================================================
        h_d, w_d = disparity_crop.shape
        left_area = disparity_crop[:, : int(w_d * 0.35)]
        right_area = disparity_crop[:, int(w_d * 0.65) :]

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

        perception_data = {
            "rect_l": rect_l,
            "rect_r": rect_r,
            "disp_color": disp_color,
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

    panel_w, panel_h = 640, 360

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        data = perc.process(frame_l, frame_r)

        # 1. 左カメラ映像 (左上)
        img_left = cv2.resize(data["rect_l"], (panel_w, panel_h))
        cv2.putText(
            img_left,
            "Left Camera",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # 2. 右カメラ映像 (右上)
        img_right = cv2.resize(data["rect_r"], (panel_w, panel_h))
        cv2.putText(
            img_right,
            "Right Camera",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # 3. 視差マップ / 深度 (左下)
        img_disp = cv2.resize(data["disp_color"], (panel_w, panel_h))
        cv2.putText(
            img_disp,
            "Disparity Map (Depth)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 4. 認識オーバーレイ結果 (右下)
        result_img = data["full_overlay"]
        img_result = cv2.resize(result_img, (panel_w, panel_h))
        cv2.putText(
            img_result,
            f"Bias: {data['stereo_bias']:.2f} YawErr: {data['wall_yaw_error']:.2f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 2x2 グリッド合成
        top_row = np.hstack((img_left, img_right))
        bottom_row = np.hstack((img_disp, img_result))
        grid_view = np.vstack((top_row, bottom_row))

        cv2.imshow("Perception Multi-Debug", grid_view)

        if cv2.waitKey(1) == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()