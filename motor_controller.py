"""
MotorController (Actuator Driver Only)
ステアリングおよびスロットル指令 (-1.0 ~ +1.0) を PWM パルス幅に変換して出力する専用モジュール。
"""
import time

ADAFRUIT_AVAILABLE = False
PCA9685 = None
board = None
busio = None

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    ADAFRUIT_AVAILABLE = True
except (ImportError, NotImplementedError, Exception) as e:
    print(f"情報: PCA9685ライブラリ無効 ({e})。シミュレーション/モックモードで動作します。")
    ADAFRUIT_AVAILABLE = False


class MotorController:
    def __init__(self,
                 steer_ch: int = 0,
                 steer_center: int = 300,
                 steer_range: int = 150,     # center ± range (150 ~ 450)
                 throttle_ch: int = 1,
                 throttle_neutral: int = 307, # ESCニュートラル (約1.5ms)
                 throttle_range: int = 100):  # 前進/後退の最大変化幅
        
        self.steer_ch = steer_ch
        self.steer_center = steer_center
        self.steer_min = steer_center - steer_range
        self.steer_max = steer_center + steer_range
        
        self.throttle_ch = throttle_ch
        self.throttle_neutral = throttle_neutral
        self.throttle_min = throttle_neutral - throttle_range
        self.throttle_max = throttle_neutral + throttle_range

        self.pca = None
        if ADAFRUIT_AVAILABLE and (busio is not None) and (board is not None):
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(i2c)
                self.pca.frequency = 50  # 50Hz (サーボ / ESC 周期: 20ms)
                time.sleep(0.1)
                self.stop()
                print("MotorController: PCA9685 の初期化に成功しました。")
            except Exception as e:
                print(f"MotorController エラー: PCA9685初期化失敗 -> {e}")
                self.pca = None

    def _set_pwm(self, channel: int, pulse: int) -> None:
        """12-bit パルス値 (0-4095) を 16-bit デューティサイクルに変換して出力"""
        if self.pca is not None:
            duty = int(pulse * 65535 / 4096)
            self.pca.channels[channel].duty_cycle = max(0, min(65535, duty))

    def drive(self, steer: float, throttle: float = 0.0) -> None:
        """
        モータおよびサーボへの出力
        :param steer: ステアリング指令 (-1.0: 最大左 〜 0.0: センター 〜 +1.0: 最大右)
        :param throttle: スロットル指令 (-1.0: 最大後退/ブレーキ 〜 0.0: 停止 〜 +1.0: 最大前進)
        """
        # 入力を [-1.0, 1.0] に正規化クリッピング
        steer_clamped = max(-1.0, min(1.0, float(steer)))
        throttle_clamped = max(-1.0, min(1.0, float(throttle)))

        # パルス幅へマッピング
        steer_pulse = int(self.steer_center + steer_clamped * (self.steer_max - self.steer_center))
        throttle_pulse = int(self.throttle_neutral + throttle_clamped * (self.throttle_max - self.throttle_neutral))

        # ハードウェアリミット保護
        steer_pulse = max(self.steer_min, min(self.steer_max, steer_pulse))
        throttle_pulse = max(self.throttle_min, min(self.throttle_max, throttle_pulse))

        # 実機へ出力
        self._set_pwm(self.steer_ch, steer_pulse)
        self._set_pwm(self.throttle_ch, throttle_pulse)

        print(f"\r[Actuator] Steer: {steer_clamped:+1.2f} (Pulse:{steer_pulse:3d}) | "
              f"Throttle: {throttle_clamped:+1.2f} (Pulse:{throttle_pulse:3d})", end="", flush=True)

    def stop(self) -> None:
        """ステアリングを中立に戻し、スロットルをニュートラルにする"""
        self._set_pwm(self.steer_ch, self.steer_center)
        self._set_pwm(self.throttle_ch, self.throttle_neutral)

    def cleanup(self) -> None:
        """終了時の安全停止処理"""
        print("\nアクチュエータを停止・中立に復帰しています...")
        self.stop()
        time.sleep(0.1)


if __name__ == "__main__":
    print("MotorController 単体テスト開始 (ドライバ動作確認)")
    motor = MotorController()
    try:
        print("\n1. センター & 停止")
        motor.drive(steer=0.0, throttle=0.0)
        time.sleep(0.5)

        print("\n2. 左にフルステア、微速前進")
        motor.drive(steer=-1.0, throttle=0.2)
        time.sleep(0.5)

        print("\n3. 右にフルステア、微速前進")
        motor.drive(steer=1.0, throttle=0.2)
        time.sleep(0.5)
    finally:
        motor.cleanup()