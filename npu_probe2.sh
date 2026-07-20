#!/bin/bash
# =============================================================================
# npu_probe2.sh — lève les ambiguïtés du premier probe.
#   bash npu_probe2.sh 2>&1 | tee npu_probe2_$(date +%Y%m%d).txt
# =============================================================================
sep() { echo; echo "=== $* ==================================================="; }

sep "A. Identification exacte du SoC (pour la section Hardware de l'article)"
cat /sys/devices/soc0/machine 2>/dev/null
cat /sys/devices/soc0/family 2>/dev/null
cat /sys/devices/soc0/soc_id 2>/dev/null
cat /sys/devices/soc0/revision 2>/dev/null
cat /sys/devices/soc0/hw_platform 2>/dev/null
tr -d '\0' < /sys/firmware/devicetree/base/compatible 2>/dev/null; echo
tr -d '\0' < /sys/firmware/devicetree/base/qcom,msm-id 2>/dev/null | xxd | head -2

sep "B. Cœurs CPU : combien sont réellement en ligne ?"
echo "nproc            : $(nproc)"
echo "present          : $(cat /sys/devices/system/cpu/present)"
echo "online           : $(cat /sys/devices/system/cpu/online)"
echo "offline          : $(cat /sys/devices/system/cpu/offline)"
grep -c ^processor /proc/cpuinfo
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  n=$(basename $c)
  echo "$n online=$(cat $c/online 2>/dev/null || echo 'always') \
maxfreq=$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done

sep "C. Gouverneurs disponibles (il FAUT figer en performance pour les benchmarks)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null
echo "--- pour figer, lors des runs de mesure :"
echo "    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > \$g; done"

sep "D. FastRPC : le device ne s'appelle pas 'fastrpc' sur ce kernel Android"
ls -la /dev | grep -iE 'rpc|dsp|adsp|cdsp|smd|glink' || echo "aucun"
ls /sys/class/misc 2>/dev/null | grep -iE 'rpc|dsp'

sep "E. debugfs monté ? (le premier probe ne pouvait pas conclure — pipe masquait l'erreur)"
mount | grep -w debugfs || echo "debugfs NON monté -> essayer: mount -t debugfs none /sys/kernel/debug"
ls -d /sys/kernel/debug/clk 2>/dev/null && \
  { echo "--- horloges NSP/CDSP ---"; grep -iE 'nsp|cdsp|q6|turing|hmx|hvx' /sys/kernel/debug/clk/clk_summary | head -20; } \
  || echo "/sys/kernel/debug/clk absent"

sep "F. Le rail 'battery' réagit-il à la charge ? (=> énergie par inférence)"
rd() { echo "$(( $(cat /sys/class/power_supply/battery/current_now) ))uA \
$(( $(cat /sys/class/power_supply/battery/voltage_now) ))uV"; }
echo "status : $(cat /sys/class/power_supply/battery/status 2>/dev/null)"
echo "idle   : $(rd)"; sleep 2; echo "idle   : $(rd)"
echo "-- charge CPU 6 threads pendant 8 s --"
for i in $(seq 1 6); do (timeout 8 bash -c 'while :; do :; done') & done
sleep 4; echo "load   : $(rd)"
sleep 3; echo "load   : $(rd)"
wait 2>/dev/null
sleep 3; echo "après  : $(rd)"
echo ">>> si le courant varie de plus de ~20 mA entre idle et load, le rail est exploitable."

sep "G. iio (autres ADC de mesure éventuels)"
for d in /sys/bus/iio/devices/*; do
  [ -e "$d" ] || continue
  echo "$(basename $d): $(cat $d/name 2>/dev/null)"
  ls $d | grep -E 'voltage|current|power' | head -5
done

sep "H. Options réellement acceptées par le delegate (liste complète triée)"
DEL="${QAIRT:-/opt/qcom/qairt-new/qairt/2.48.0.260626}/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"
strings "$DEL" | grep -iE 'Invalid|Unsupported|unknown' | grep -i option | sort -u | head -20

sep "I. Outils QNN : versions et aide (pour le pipeline tflite -> QNN)"
export LD_LIBRARY_PATH=/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2:$LD_LIBRARY_PATH
/opt/qcom/qirp-sdk/bin/aarch64-oe-linux-gcc11.2/qnn-net-run --version 2>&1 | head -3
/opt/qcom/qirp-sdk/bin/aarch64-oe-linux-gcc11.2/qnn-throughput-net-run --help 2>&1 | head -25
ls /opt/qcom/qirp-sdk/bin/aarch64-oe-linux-gcc11.2/ | grep -iE 'convert|lib-generator|quantiz'

echo; echo "=== DONE ==="