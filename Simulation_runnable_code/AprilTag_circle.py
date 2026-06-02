import time
import math
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

def move_tag_via_service():
    # 1. 初始化节点
    node = Node()
    
    # 对应你测试成功的服务路径
    model_name = "my_apriltag"
    service = "/world/default/set_pose"
    
    print(f"🚀 动态靶标系统已启动：正在通过 Service 控制 [{model_name}]...")
    
    # 2. 运动参数
    center_x, center_y = 0.0, 0.0  # 圆心位置
    radius = 5                   # 旋转半径
    omega = 0.1                    # 旋转角速度 (rad/s)
    start_time = time.time()

    try:
        while True:
            t = time.time() - start_time
            
            # 3. 计算位置
            curr_x = center_x + radius * math.cos(omega * t)
            curr_y = center_y + radius * math.sin(omega * t)
            
            # 4. 构建请求消息
            req = Pose()
            req.name = model_name
            req.position.x = curr_x
            req.position.y = curr_y
            req.position.z = 0.05
            req.orientation.w = 1.0
            
            # 5. 调用服务 (Request 模式)
            # 参数: 服务名, 请求消息, 请求类型, 响应类型, 超时时间(ms)
            # 注意：服务调用比话题发布更耗资源，频率不建议超过 30Hz
            node.request(service, req, Pose, Boolean, 500)
            
            # 控制更新频率
            time.sleep(0.04) # 约 25Hz
            
    except KeyboardInterrupt:
        print("\n🛑 动态靶标停止")

if __name__ == "__main__":
    move_tag_via_service()