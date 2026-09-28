import time

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    ADAFRUIT_AVAILABLE = True
except ImportError as e:
    print(f"警告: Adafruitライブラリの読み込みに失敗しました ({e})。モックモードで実行します。")
    ADAFRUIT_AVAILABLE = False

class MotorController:
    def __init__(self, channel=0, center=300, min_pulse=150, max_pulse=450,
                 kp=0.0, kc=0.0, ks=2.5, kyaw=0.0):
        self.channel = channel
        self.center = center
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        
        # ゲイン
        self.kp = kp
        self.kc = kc
        self.ks = ks
        self.kyaw = kyaw
        
        self.pca = None
        if ADAFRUIT_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(i2c)
                self.pca.frequency = 50  
                time.sleep(0.1)
                self._set_pwm(self.center)
                print("MotorController: PCA9685 の初期化に成功しました。")
            except Exception as e:
                print(f"MotorController エラー: PCA9685の初期化に失敗しました -> {e}")

    def _set_pwm(self, pulse):
        if self.pca is not None:
            duty = int(pulse * 65535 / 4096)
            self.pca.channels[self.channel].duty_cycle = duty

    def drive(self, offset, curvature, stereo_bias, wall_yaw_error):
        pulse_change = (self.kp * offset) + (self.kc * curvature) + (self.ks * stereo_bias) + (self.kyaw * wall_yaw_error)
        target_pulse = int(self.center + pulse_change)
        target_pulse = max(self.min_pulse, min(self.max_pulse, target_pulse))
        
        self._set_pwm(target_pulse)
        print(f"\r[Motor] Offset: {offset:+4d} | Curv: {curvature:+.4f} | ContBias: {stereo_bias:+.1f} | YawErr: {wall_yaw_error:+.1f} | Pulse: {target_pulse}", end="", flush=True)

    def cleanup(self):
        if self.pca is not None:
            print("\nステアリングをセンターに復帰しています...")
            self._set_pwm(self.center)
            time.sleep(0.2)


if __name__ == "__main__":
    # モータコントローラ単体デバッグ用
    print("モータコントローラ単体テストを開始します...")
    motor = MotorController()
    try:
        # 左へ切る
        print("Left...")
        motor.drive(offset=-50, curvature=0, stereo_bias=0, wall_yaw_error=0)
        time.sleep(1)
        # 右へ切る
        print("\nRight...")
        motor.drive(offset=50, curvature=0, stereo_bias=0, wall_yaw_error=0)
        time.sleep(1)
        # センター
        print("\nCenter...")
        motor.drive(offset=0, curvature=0, stereo_bias=0, wall_yaw_error=0)
        time.sleep(1)
    finally:
        motor.cleanup()