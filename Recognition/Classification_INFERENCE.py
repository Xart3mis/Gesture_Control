import serial
import pickle
import numpy as np
from time import sleep
from scipy.interpolate import interp1d

print("Loading Model and Labels")
pickle_f = pickle.load(open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "rb"))

labels = pickle_f[1]
Tnew = pickle_f[0]
clf = pickle_f[2]

MIN_GESTURE_LEN = 5


class SerialIMU:
    def __init__(self, port: str, baud_rate: int) -> None:
        self.pyb = serial.Serial(port, baud_rate, timeout=1)
        self.gesture = []

    def update(self) -> None:
        h, p, r, ax, ay, az, mx, my, mz = [], [], [], [], [], [], [], [], []
        while self.pyb.in_waiting > 8:
            line = self.pyb.readline().decode("utf-8").strip()
            sleep(0.06)
            print("reading gesture... ", end="      \r")
            # print(line)
            line_arr = line.split(",")
            try:
                h.append(float(line_arr[0]))
                p.append(float(line_arr[1]))
                r.append(float(line_arr[2]))
                ax.append(float(line_arr[3]))
                ay.append(float(line_arr[4]))
                az.append(float(line_arr[5]))
                mx.append(float(line_arr[6]))
                my.append(float(line_arr[7]))
                mz.append(float(line_arr[8]))

            except ValueError as e:
                if "string to float" in str(e).lower():
                    print("something went wrong on embedded side")
                    return
                print(line, e)
        if len(h) > MIN_GESTURE_LEN:
            print("yadin omy")
            self.gesture.append([h, p, r, ax, ay, az])

            for i, v in enumerate(self.gesture):
                Told = np.arange(0, len(v[0]))
                F = interp1d(Told, v, fill_value="extrapolate")
                self.gesture[i] = F(Tnew)


if __name__ == "__main__":
    simu = SerialIMU("/dev/ttyACM0", 115200)
    print("Ready")
    while True:
        simu.update()

        if len(simu.gesture) > 0:
            if len(simu.gesture[0]) > MIN_GESTURE_LEN:
                print("WHAT")
                gesture = np.asarray(simu.gesture)
                print(labels[clf.predict(gesture)[0]])

            simu.gesture.clear()
