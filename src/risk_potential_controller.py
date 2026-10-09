"""
risk_potential_controller.py
東京農工大学 ポンサトーン研究室のリスクポテンシャル場（Dynamic APF）
および前方注視ドライバモデルに基づく操舵コントローラ
"""
import math
import numpy as np


class RiskPotentialController:
    def __init__(self,
                 k_lane: float = 0.003,   # 引力ゲイン (車線中央への復帰力)
                 k_curv: float = 120.0,   # 曲率先行予測ゲイン (0.5 * v^2 * tau^2 に相当)
                 k_wall: float = 0.6,     # 斥力振幅ゲイン (壁面接近時の最大反発力)
                 k_yaw: float = 0.35,     # ヨー角ダンピングゲイン (ハンチング・蛇行抑制)
                 b_safe: float = 0.25,    # 斥力不感帯 (安全クリアランス閾値)
                 sigma_w: float = 0.15):  # 斥力の急峻度 (指数関数の減衰幅)
        self.k_lane = float(k_lane)
        self.k_curv = float(k_curv)
        self.k_wall = float(k_wall)
        self.k_yaw = float(k_yaw)
        self.b_safe = float(b_safe)
        self.sigma_w = float(sigma_w)

    def compute(self, offset: float, curvature: float,
                stereo_bias: float, wall_yaw_error: float):
        """
        センサ入力からステアリング指令 (-1.0 ~ +1.0) を算出
        
        :param offset: 走路中心からの横変位偏差 [px]
        :param curvature: 走行ラインの曲率 [1/px または 1/m]
        :param stereo_bias: 左右壁面クリアランスの非対称度 (正: 右壁接近, 負: 左壁接近)
        :param wall_yaw_error: 壁面に対する機体の傾き角 [rad]
        :return: (steer_cmd, f_att, f_rep, f_yaw)
        """
        # 1. 引力ポテンシャル勾配 F_att (前方注視モデルによる走路追従)
        y_preview_err = offset + (self.k_curv * curvature)
        f_att = self.k_lane * y_preview_err

        # 2. 斥力ポテンシャル勾配 F_rep (指数関数型リスク場)
        f_rep = 0.0
        abs_bias = abs(stereo_bias)
        if abs_bias > self.b_safe:
            exponent = min((abs_bias - self.b_safe) / self.sigma_w, 4.0)
            rep_mag = self.k_wall * (math.exp(exponent) - 1.0)
            sign = 1.0 if stereo_bias > 0 else -1.0
            f_rep = sign * rep_mag

        # 3. ヨー角ダンピング項 F_yaw (蛇行抑制・姿勢安定化)
        f_yaw = self.k_yaw * wall_yaw_error

        # 4. 合成操舵指令値 (-1.0 ~ +1.0 に飽和制限)
        steer_cmd = f_att + f_rep + f_yaw
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

        return steer_cmd, f_att, f_rep, f_yaw


if __name__ == "__main__":
    # コントローラ単体動作確認テスト
    ctrl = RiskPotentialController()
    print("RiskPotentialController 単体テスト:")
    cmd, att, rep, yaw = ctrl.compute(offset=-30.0, curvature=0.001, stereo_bias=0.4, wall_yaw_error=0.1)
    print(f"Steer: {cmd:+.2f} | Att: {att:+.2f} | Rep: {rep:+.2f} | Yaw: {yaw:+.2f}")