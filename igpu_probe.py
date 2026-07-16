# -*- coding: utf-8 -*-
import openvino as ov
core = ov.Core()
print("available_devices:", core.available_devices)
for d in core.available_devices:
    print(f"\n== {d} ==")
    for key in ("FULL_DEVICE_NAME", "DEVICE_TYPE", "DEVICE_ARCHITECTURE", "GPU_DEVICE_ID"):
        try:
            print(f"  {key}: {core.get_property(d, key)}")
        except Exception as e:
            print(f"  {key}: <n/a: {str(e)[:50]}>")

# 專門找 Intel iGPU：逐一列舉 GPU.0 / GPU.1
print("\n== 逐一探測 GPU.x ==")
for i in range(4):
    dev = f"GPU.{i}"
    try:
        name = core.get_property(dev, "FULL_DEVICE_NAME")
        dtype = core.get_property(dev, "DEVICE_TYPE")
        print(f"  {dev}: {name} | type={dtype}")
    except Exception as e:
        print(f"  {dev}: <不存在>")
