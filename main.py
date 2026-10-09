"""
main.py
ステレオ視覚誘導 TT-02 リアルタイム走行制御
"""
import cv2
import time
import sys

# 各モジュールのインポート
from perception import Perception, gstreamer_pipeline
from trajectory_planner import TrajectoryPlanner
from risk_potential_controller import RiskPotentialController
from motor_controller import MotorController


def main():
    print("システム初期化中...")

    # 1. 各モジュールのインスタンス化
    perception = Perception(calib_file='stereo_params.npz')
    planner = TrajectoryPlanner()
    controller = RiskPotentialController(
        k_lane=0.003,
        k_curv=120.0,
        k_wall=0.6,
        k_yaw=0.35,
        b_safe=0.25,
        sigma_w=0.15
    )
    motor = MotorController(steer_ch=0, throttle_ch=1)

    # 一定速度設定 (0.0: 停止, 1.0: 全開前進)
    CONSTANT_THROTTLE = 0.20

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
            # 左右フレームの同期ラッチ
            if not (cap_l.grab() and cap_r.grab()):
                print("\nフレームのグラブに失敗しました。")
                break

            ret_l, frame_l = cap_l.retrieve()
            ret_r, frame_r = cap_r.retrieve()
            if not ret_l or not ret_r:
                print("\nフレームのデコードに失敗しました。")
                break

            # --- A. 認識処理 ---
            p_data = perception.process(frame_l, frame_r)

            # --- B. 軌跡生成 ---
            w = p_data['rect_l'].shape[1]
            t_data = planner.calculate(
                p_data['mask_ground'], p_data['scale'],
                p_data['roi_top'], p_data['roi_bottom'], w
            )

            # --- C. リスクポテンシャル制御の計算 ---
            steer_cmd, f_att, f_rep, f_yaw = controller.compute(
                offset=t_data['offset'],
                curvature=t_data['curvature'],
                stereo_bias=p_data['stereo_bias'],
                wall_yaw_error=p_data['wall_yaw_error']
            )

            # --- D. アクチュエータ駆動 ---
            motor.drive(steer=steer_cmd, throttle=CONSTANT_THROTTLE)

            # --- E. 描画・UI更新 ---
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                output = p_data['rect_l'].copy()

                # マスクの合成
                alpha = 0.4
                roi_t, roi_b = p_data['roi_top'], p_data['roi_bottom']
                output[roi_t:roi_b, :] = cv2.addWeighted(
                    output[roi_t:roi_b, :], 1 - alpha, p_data['full_overlay'], alpha, 0
                )

                # 軌跡の描画
                if t_data['curve_pts'] is not None:
                    cv2.polylines(output, [t_data['curve_pts']], isClosed=False, color=(0, 255, 255), thickness=4)
                    cv2.circle(output, (t_data['target_x_bot'], roi_b - 10), 8, (0, 255, 255), -1)
                cv2.line(output, (t_data['x_center_frame'], roi_t), (t_data['x_center_frame'], roi_b), (255, 0, 0), 2)

                # テキスト情報描画
                cv2.putText(output, f"Steer Cmd: {steer_cmd:+.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(output, f"Att:{f_att:+.2f} Rep:{f_rep:+.2f} Yaw:{f_yaw:+.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(output, f"Offset: {t_data['offset']:+4d} px", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output, f"Curv: {t_data['curvature']:+.5f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output, f"Wall Bias: {p_data['stereo_bias']:+.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                cv2.putText(output, f"Wall Yaw: {p_data['wall_yaw_error']:+.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                cv2.imshow("Stereo Lane Centering", cv2.resize(output, (640, 360)))
                last_display_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n停止シグナルを受信しました。")
                    break

    except KeyboardInterrupt:
        print("\nCtrl+C により中断されました。")
    finally:
        motor.cleanup()
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
        print("正常にリソースを解放して終了しました。")


if __name__ == "__main__":
    main()