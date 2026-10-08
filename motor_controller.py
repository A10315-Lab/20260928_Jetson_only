"""
MotorController for Steering Control
Python 3.10 / Anaconda Compatible
"""
import math
import time

ADAFRUIT_AVAILABLE = False
PCA9685 = None
board = None
busio = None

# Python 3.10 および PC/Anaconda 環境での安全なインポート
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    ADAFRUIT_AVAILABLE = True
except (ImportError, NotImplementedError, Exception) as e:
    # PC環境やライブラリ未導入環境ではモックモードへ退避
    print(f"情報: ハードウェアライブラリが無効です ({e})。シミュレーション/モックモードで動作します。")
    ADAFRUIT_AVAILABLE = False


class MotorController:
    """
    リスクポテンシャル場（Risk Potential Field）
    および前方注視ドライバモデルに基づく操舵サーボコントローラ
    """
    def __init__(self, channel: int = 0, center: int = 300, 
                 min_pulse: int = 150, max_pulse: int = 450,
                 kp: float = 0.35, kc: float = 500.0, 
                 ks: float = 25.0, kyaw: float = 15.0,
                 b_safe: float = 0.25, sigma_w: float = 0.15):
        
        self.channel = int(channel)
        self.center = int(center)
        self.min_pulse = int(min_pulse)
        self.max_pulse = int(max_pulse)
        
        # --- リスクポテンシャル法パラメータ ---
        self.k_lane = float(kp)       # 引力ゲイン（中心線追従力）
        self.k_curv = float(kc)       # 前方注視・カーブ先読みシフト量ゲイン
        self.k_wall = float(ks)       # 斥力ゲイン（壁面反発力振幅）
        self.k_yaw = float(kyaw)      # ダンピング項（車体姿勢の蛇行抑制）
        
        self.b_safe = float(b_safe)   # 斥力不感帯
        self.sigma_w = float(sigma_w) # 斥力場の急峻度

        # 表示・外部参照用
        self.kp = kp
        self.kc = kc
        self.ks = ks
        self.kyaw = kyaw
        
        self.pca = None
        if ADAFRUIT_AVAILABLE and (busio is not None) and (board is not None):
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(i2c)
                self.pca.frequency = 50  
                time.sleep(0.1)
                self._set_pwm(self.center)
                print("MotorController: PCA9685 の初期化に成功しました。")
            except Exception as e:
                print(f"MotorController エラー: PCA9685の初期化に失敗しました -> {e}")
                self.pca = None

    def _set_pwm(self, pulse: int) -> None:
        if self.pca is not None:
            # 12-bit (0-4095) から 16-bit (0-65535) へのデューティサイクル変換
            duty = int(pulse * 65535 / 4096)
            self.pca.channels[self.channel].duty_cycle = duty

    def compute_potential_steering(self, offset: float, curvature: float, 
                                   stereo_bias: float, wall_yaw_error: float):
        """
        ポテンシャル勾配から操舵パルス変化量を算出
        """
        # 1. 引力ポテンシャル勾配 F_att (前方注視モデルによる車線追従)
        preview_offset = offset + (self.k_curv * curvature)
        f_att = self.k_lane * preview_offset

        # 2. 斥力ポテンシャル勾配 F_rep (指数関数型リスク場)
        f_rep = 0.0
        abs_bias = abs(stereo_bias)
        if abs_bias > self.b_safe:
            # 指数発散防止のために指数部をクリッピング (上限 4.0)
            exponent = min((abs_bias - self.b_safe) / self.sigma_w, 4.0)
            rep_mag = self.k_wall * (math.exp(exponent) - 1.0)
            sign = 1.0 if stereo_bias > 0 else -1.0
            f_rep = sign * rep_mag

        # 3. ヨー角ダンピング項 F_yaw (蛇行抑制)
        f_yaw = self.k_yaw * wall_yaw_error

        # 4. 合成操舵補正パルス量
        pulse_change = f_att + f_rep + f_yaw
        return pulse_change, f_att, f_rep, f_yaw

    def drive(self, offset: float, curvature: float, 
              stereo_bias: float, wall_yaw_error: float) -> None:
        pulse_change, f_att, f_rep, f_yaw = self.compute_potential_steering(
            offset, curvature, stereo_bias, wall_yaw_error
        )
        
        target_pulse = int(round(self.center + pulse_change))
        # ハードウェアリミッター（サーボの可動範囲内に収める）
        target_pulse = max(self.min_pulse, min(self.max_pulse, target_pulse))
        
        self._set_pwm(target_pulse)
        print(f"\r[Motor] Att:{f_att:+5.1f} | Rep:{f_rep:+5.1f} | Yaw:{f_yaw:+5.1f} | Pulse:{target_pulse:4d}", end="", flush=True)

    def cleanup(self) -> None:
        if self.pca is not None:
            print("\nステアリングをセンターに復帰しています...")
            self._set_pwm(self.center)
            time.sleep(0.2)


if __name__ == "__main__":
    print("Python 3.10 / Anaconda コンパイルテスト完了")
    motor = MotorController()
    try:
        motor.drive(offset=-50.0, curvature=0.0, stereo_bias=0.0, wall_yaw_error=0.0)
        time.sleep(0.5)
        motor.drive(offset=0.0, curvature=0.0, stereo_bias=0.6, wall_yaw_error=0.0)
        time.sleep(0.5)
        motor.drive(offset=0.0, curvature=0.0, stereo_bias=0.0, wall_yaw_error=0.0)
    finally:
        motor.cleanup()