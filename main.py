import cv2
import time
import sys

# 各モジュールのインポート
from perception import Perception, gstreamer_pipeline
from trajectory_planner import TrajectoryPlanner
from motor_controller import MotorController

def main():
    print("システム初期化中...")
    
    # 1. 各モジュールのインスタンス化
    perception = Perception(calib_file='stereo_params.npz')
    planner = TrajectoryPlanner()
    motor = MotorController(kp=0.0, kc=0.0, ks=2.5, kyaw=0.0) # ★ ここでゲイン調整が可能

    # 2. カメラの起動
    cap_l = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("エラー: 左右どちらかのカメラを開けませんでした。")
        sys.exit(1)

    print("リアルタイム追従を開始します。'q' キーで終了します。")
    TARGET_DISPLAY_FPS = 15
    display_interval = 1.0 / TARGET_DISPLAY_FPS
    last_display_time = time.time()

    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                print("\nフレームの取得に失敗しました。")
                break

            # --- A. 認識処理 ---
            p_data = perception.process(frame_l, frame_r)

            # --- B. 軌跡生成 ---
            w = p_data['rect_l'].shape[1]
            t_data = planner.calculate(
                p_data['mask_ground'], p_data['scale'], 
                p_data['roi_top'], p_data['roi_bottom'], w
            )

            # --- C. モータ制御 ---
            motor.drive(
                t_data['offset'], t_data['curvature'], 
                p_data['stereo_bias'], p_data['wall_yaw_error']
            )

            # --- D. 描画・UI更新 ---
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                output = p_data['rect_l'].copy()
                
                # マスクの合成
                alpha = 0.4
                roi_t, roi_b = p_data['roi_top'], p_data['roi_bottom']
                output[roi_t:roi_b, :] = cv2.addWeighted(
                    output[roi_t:roi_b, :], 1 - alpha, p_data['full_overlay'], alpha, 0)
                
                # 軌跡の描画
                if t_data['curve_pts'] is not None:
                    cv2.polylines(output, [t_data['curve_pts']], isClosed=False, color=(0, 255, 255), thickness=4)
                    cv2.circle(output, (t_data['target_x_bot'], roi_b - 10), 8, (0, 255, 255), -1)
                cv2.line(output, (t_data['x_center_frame'], roi_t), (t_data['x_center_frame'], roi_b), (255, 0, 0), 2)

                # テキスト情報描画
                cv2.putText(output, f"Offset: {t_data['offset']:+4d} px", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output, f"Curv: {t_data['curvature']:+.5f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output, f"Wall Cont. Bias: {p_data['stereo_bias']:+.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(output, f"Wall Yaw Err: {p_data['wall_yaw_error']:+.2f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                cv2.imshow("Stereo Lane Centering", cv2.resize(output, (640, 360)))
                last_display_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n停止シグナルを受信しました。")
                break

    finally:
        motor.cleanup()
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
        print("正常にリソースを解放して終了しました。")

if __name__ == "__main__":
    main()