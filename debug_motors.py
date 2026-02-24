import serial
from scservo_sdk import PortHandler, PacketHandler

def scan(name, path):
    print(f"\n--- 正在扫描 {name} ({path}) ---")
    port = PortHandler(path)
    packet = PacketHandler(1.0)
    
    if not port.openPort():
        print(f"  🚫 错误：无法打开端口 {path}，请检查权限或设备是否存在")
        return

    # 设置波特率 1M (飞特舵机默认)
    port.setBaudRate(1000000)
    found_any = False
    
    for i in range(1, 11):
        model, res, err = packet.ping(port, i)
        if res == 0:
            print(f"  ✅ [ID:{i}] 响应正常 (型号: {model})")
            found_any = True
            
    if not found_any:
        print("  ❌ 未发现任何舵机。请检查：1. 12V电源是否开启 2. 信号线是否插紧")
    
    port.closePort()

scan("从机 (Follower)", "/dev/follower_arm")
scan("主机 (Leader)", "/dev/leader_arm")
