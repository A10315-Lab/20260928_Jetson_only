import cv2
import numpy as np

class TrajectoryPlanner:
    def __init__(self):
        self.lpf_kernel = np.ones(5) / 5.0

    def calculate(self, mask_ground, scale, roi_top, roi_bottom, w):
        proc_h = mask_ground.shape[0]
        x_center_frame = w // 2
        
        y_pts_buf = np.empty(proc_h, dtype=np.float32)
        x_pts_buf = np.empty(proc_h, dtype=np.float32)
        valid_count = 0
        
        min_road_width_px = 40
        max_x_jump_px = 40
        consecutive_limit = 8
        missing_count = 0
        y_cutoff_real = roi_top
        last_x = None

        for y in range(proc_h - 1, -1, -1):
            if y < int(proc_h * 0.25):
                y_cutoff_real = int(y / scale) + roi_top
                break

            x_indices = np.where(mask_ground[y, :] > 0)[0]
            if len(x_indices) > 0:
                splits = np.where(np.diff(x_indices) > 1)[0] + 1
                clusters = np.split(x_indices, splits)
                
                if last_x is None:
                    best_cluster = max(clusters, key=len)
                else:
                    valid_clusters = [c for c in clusters if abs(np.mean(c) - last_x) <= max_x_jump_px]
                    best_cluster = max(valid_clusters, key=len) if valid_clusters else []

                if len(best_cluster) >= min_road_width_px:
                    x_mean = np.mean(best_cluster)
                    last_x = x_mean
                    missing_count = 0
                    y_pts_buf[valid_count] = int(y / scale) + roi_top
                    x_pts_buf[valid_count] = int(x_mean / scale)
                    valid_count += 1
                else:
                    missing_count += 1
            else:
                missing_count += 1
                
            if missing_count >= consecutive_limit:
                y_cutoff_real = int(y / scale) + roi_top
                break

        offset = 0
        curvature = 0.0
        curve_pts = None
        target_x_bot = x_center_frame

        if valid_count >= 5:
            y_pts = y_pts_buf[:valid_count]
            x_pts = x_pts_buf[:valid_count]
            x_lpf = np.convolve(x_pts, self.lpf_kernel, mode='same')
            
            dev = np.abs(x_pts - x_lpf)
            stability_weights = np.exp(-0.5 * (dev / 15.0) ** 2)
            y_norm = (y_pts - y_cutoff_real) / max(1, (roi_bottom - y_cutoff_real))
            final_weights = ((y_norm ** 2) + 0.05) * stability_weights
            
            anchor_y = np.array([roi_bottom, roi_bottom - 4, roi_bottom - 8], dtype=np.float32)
            anchor_x = np.array([x_center_frame, x_center_frame, x_center_frame], dtype=np.float32)
            anchor_weights = np.array([200.0, 100.0, 50.0], dtype=np.float32)
            
            y_pts_fixed = np.concatenate([y_pts, anchor_y])
            x_lpf_fixed = np.concatenate([x_lpf, anchor_x])
            final_weights_fixed = np.concatenate([final_weights, anchor_weights])
            
            poly = np.polyfit(y_pts_fixed - roi_bottom, x_lpf_fixed, 2, w=final_weights_fixed)
            
            plot_y = np.linspace(roi_bottom, y_cutoff_real, 30)
            plot_x = np.polyval(poly, plot_y - roi_bottom)
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            target_x_bot = int(np.polyval(poly, 0))
            offset = target_x_bot - x_center_frame
            curvature = poly[0]

        traj_data = {
            'offset': offset, 'curvature': curvature, 
            'curve_pts': curve_pts, 'target_x_bot': target_x_bot,
            'x_center_frame': x_center_frame, 'y_cutoff_real': y_cutoff_real
        }
        return traj_data

if __name__ == "__main__":
    # 軌跡生成部単体デバッグ用
    print("軌跡生成部のデバッグを開始します")
    planner = TrajectoryPlanner()
    
    # テスト用のダミーマスク生成 (幅320, 高さ150くらいの解像度を想定)
    dummy_mask = np.zeros((150, 320), dtype=np.uint8)
    cv2.line(dummy_mask, (160, 150), (200, 50), 255, 60) # 右に曲がる道
    
    res = planner.calculate(dummy_mask, scale=0.5, roi_top=300, roi_bottom=600, w=1280)
    print(f"計算結果: Offset={res['offset']}, Curvature={res['curvature']:.4f}")