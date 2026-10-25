#!/system/bin/sh
exec > /data/local/tmp/tf_out.txt 2>&1
# Smoke test for Google Device Run: who am I, which GPU, can I write and run in /data/local/tmp, is uptime sane.
echo "PROBE model=$(getprop ro.product.model) soc=$(getprop ro.soc.model) board=$(getprop ro.board.platform) sdk=$(getprop ro.build.version.sdk)"
echo "PROBE id=$(id) pwd=$(pwd)"
echo "PROBE gpu=$(cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null) egl=$(getprop ro.hardware.egl) vulkan=$(getprop ro.hardware.vulkan)"
ls /vendor/lib64/libOpenCL.so /system/vendor/lib64/libOpenCL.so 2>&1 | sed 's/^/PROBE opencl: /'
u1=$(cut -d' ' -f1 /proc/uptime); sleep 2; u2=$(cut -d' ' -f1 /proc/uptime); echo "PROBE uptime_sleep2=$u1..$u2"
echo "PROBE tmp: $(ls -ld /data/local/tmp) df: $(df -h /data/local/tmp | tail -1)"
