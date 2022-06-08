import time
import json
import serial
import pickle
import numpy as np
from rdp import rdp
from tslearn import utils

print("Loading Model and Labels")
pickle_f = pickle.load(open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "rb"))

labels = pickle_f[1]
clf = pickle_f[0]


class SerialIMU:
    def __init__(self, port: str, baud_rate: int) -> None:
        self.pyb = serial.Serial(port, baud_rate)
        self.gesture = []

    def update(self) -> None:
        while self.pyb.in_waiting > 8:
            line = self.pyb.readline().decode("utf-8").strip()
            print("reading gesture... ", end="      \r")
            # print(line)
            line_arr = line.split(",")
            self.gesture.append(
                [
                    float(line_arr[0]),
                    float(line_arr[1]),
                    float(line_arr[2]),
                    float(line_arr[3]),
                    float(line_arr[4]),
                    float(line_arr[5]),
                    float(line_arr[6]),
                    float(line_arr[7]),
                    float(line_arr[8]),
                ]
            )
            time.sleep(0.05)


if __name__ == "__main__":
    simu = SerialIMU("/dev/ttyACM0", 115200)
    print("Ready")
    while True:
        simu.update()

        if len(simu.gesture) > 0:
            if len(simu.gesture) > 10:
                gesture = np.array([simu.gesture])
                if len(gesture.shape) == 3:
                    if gesture.shape[2] == 9:
                        print("Recognizing gesture...")
                        print(labels[clf.predict(utils.to_time_series(rdp(gesture, epsilon=0.4)))[0]])

        simu.gesture.clear()
