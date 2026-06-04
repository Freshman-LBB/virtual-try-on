import cv2 as cv
import numpy as np

# 检查OpenCL可用性
print("OpenCL available:", cv.ocl.haveOpenCL())

if cv.ocl.haveOpenCL():
    # 启用OpenCL
    cv.ocl.setUseOpenCL(True)
    print("OpenCL is enabled:", cv.ocl.useOpenCL())

    # 获取默认设备信息
    device = cv.ocl.Device.getDefault()
    print("\n--- OpenCL 设备信息 ---")
    print("设备名称:", device.name())

    # 检查设备类型 (使用名称中的关键词判断)
    device_name = device.name().lower()
    if "gpu" in device_name or "geforce" in device_name or "radeon" in device_name:
        device_type = "GPU"
    elif "cpu" in device_name or "processor" in device_name or "intel" in device_name:
        device_type = "CPU"
    else:
        device_type = "未知"
    print("设备类型:", device_type)

    # 其他设备信息
    try:
        print("设备供应商:", device.vendorName())
    except:
        print("设备供应商: 无法获取")

    try:
        print("驱动程序版本:", device.driverVersion())
    except:
        print("驱动程序版本: 无法获取")

    try:
        print("OpenCL版本:", device.version())
    except:
        print("OpenCL版本: 无法获取")

    try:
        print("计算单元数量:", device.maxComputeUnits())
    except:
        print("计算单元数量: 无法获取")

    try:
        print("全局内存 (MB):", device.globalMemSize() / (1024 * 1024))
    except:
        print("全局内存: 无法获取")

    try:
        print("本地内存 (KB):", device.localMemSize() / 1024)
    except:
        print("本地内存: 无法获取")

    try:
        print("最大工作组大小:", device.maxWorkGroupSize())
    except:
        print("最大工作组大小: 无法获取")

    # 尝试简单的OpenCL操作
    print("\n--- OpenCL 测试 ---")
    try:
        # 创建测试图像
        print("创建测试图像...")
        src = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)

        # 转换为UMat并进行模糊操作
        print("转换为UMat...")
        dst = cv.UMat(src)
        print("执行OpenCL模糊操作...")
        start = cv.getTickCount()
        result = cv.blur(dst, (5, 5))
        end = cv.getTickCount()
        time_ms = (end - start) * 1000 / cv.getTickFrequency()
        print(f"操作成功完成! 耗时: {time_ms:.2f} 毫秒")

        # 执行CPU版本进行比较
        print("\n使用CPU执行相同操作进行比较...")
        cv.ocl.setUseOpenCL(False)
        start_cpu = cv.getTickCount()
        result_cpu = cv.blur(src, (5, 5))
        end_cpu = cv.getTickCount()
        time_cpu_ms = (end_cpu - start_cpu) * 1000 / cv.getTickFrequency()
        print(f"CPU操作完成! 耗时: {time_cpu_ms:.2f} 毫秒")
        print(f"加速比: {time_cpu_ms / time_ms:.2f}x")

        # 重新启用OpenCL
        cv.ocl.setUseOpenCL(True)

    except Exception as e:
        print("OpenCL操作失败:", str(e))

    # 测试DNN模块的OpenCL兼容性
    print("\n--- DNN模块OpenCL测试 ---")
    try:
        # 创建一个小的测试网络
        print("创建测试网络...")
        net = cv.dnn.Net()

        # 测试设置目标为OpenCL
        print("尝试设置OpenCL目标...")
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
        print("DNN OpenCL目标设置成功! (但这不保证实际运行时不会有问题)")

        # 尝试FP16
        print("\n尝试设置OpenCL FP16目标...")
        try:
            net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)
            print("DNN OpenCL FP16目标设置成功! 这可能是一个更好的选择。")
        except:
            print("DNN OpenCL FP16目标设置失败，你的设备可能不支持FP16。")

    except Exception as e:
        print("DNN OpenCL测试失败:", str(e))
        print("请在虚拟试衣程序中使用 --target 0 (CPU目标)")
else:
    print("你的系统不支持OpenCL")

# 显示OpenCV版本
print("\nOpenCV版本:", cv.__version__)

# 显示CUDA支持情况
print("CUDA支持:", "是" if cv.cuda.getCudaEnabledDeviceCount() > 0 else "否")