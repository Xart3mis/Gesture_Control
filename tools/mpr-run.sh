cd ~/Documents/Gesture_Control
python3 tools/pyboard.py -d /dev/ttyACM0 -f cp src/main.py :
# python3 tools/pyboard.py -d /dev/ttyACM0 -f cp src/lib/*.py :lib/
python3 tools/pyboard.py -d /dev/ttyACM0 $1