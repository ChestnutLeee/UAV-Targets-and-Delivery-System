import Jetson.GPIO as GPIO
import time

# 定义引脚：这里用 BOARD 编码格式，对应物理引脚 32
SERVO_PIN = 32

def test_servo():
    # 1. 初始化设置
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    # 2. 启动 PWM，频率 50Hz
    # 计算公式：DutyCycle = (目标脉宽ms / 周期20ms) * 100
    # 0.5ms(0度) -> 2.5 | 1.5ms(90度) -> 7.5 | 2.5ms(180度) -> 12.5
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(7.5)  # 先转到中间位置
    
    print("Jetson 引脚 32 控制已就绪！")
    print("正在测试：每 2 秒切换一次位置...")

    try:
        while True:
            # 切换到 0 度位置（关闭钩子）
            print("状态：关闭")
            pwm.ChangeDutyCycle(2.5) 
            time.sleep(2)

            # 切换到 90 度位置（打开钩子）
            print("状态：打开")
            pwm.ChangeDutyCycle(7.5)
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n正在停止测试...")
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    test_servo()
