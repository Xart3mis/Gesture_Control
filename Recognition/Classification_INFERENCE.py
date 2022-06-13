import serial
import pickle
import numpy as np
from time import sleep
from scipy.interpolate import interp1d
from pyts.preprocessing import MinMaxScaler
from pyts.multivariate.transformation import MultivariateTransformer

print("Loading Model and Labels")
pickle_f = pickle.load(open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "rb"))

labels = pickle_f[0]
clf = pickle_f[1]

MIN_GESTURE_LEN = 10


class SerialIMU:
    def __init__(self, port: str, baud_rate: int) -> None:
        self.pyb = serial.Serial(port, baud_rate, timeout=1)
        self.gesture = []

    def update(self) -> None:
        h, p, r, ax, ay, az, mx, my, mz = [], [], [], [], [], [], [], [], []
        while self.pyb.in_waiting > len("0,0,0,0,0,0,1") * 5:
            line = self.pyb.readline().decode("utf-8").strip()
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
                sleep(0.08)

            except ValueError as e:
                if "string to float" in str(e).lower():
                    print("something went wrong on embedded side")
                    return
                print(line, e)

        if len(h) > 0:
            self.gesture.append([h, p, r, ax, ay, az, mx, my, mz])

    def preprocess(self) -> None:
        for i, v in enumerate(self.gesture):
            for i1, v1 in enumerate(v):
                self.gesture[i][i1] = interp1d(np.linspace(0, 99, num=len(v1)), v1, kind="cubic")(np.linspace(0, 99))
        self.gesture = MultivariateTransformer(MinMaxScaler(sample_range=(-1, 1)), flatten=False).fit_transform(
            self.gesture
        )


if __name__ == "__main__":
    simu = SerialIMU("/dev/ttyACM0", 115200)
    print("Ready")
    sleep(1.5)
    print("Waiting for gesture...")

    while True:
        simu.update()

        if len(simu.gesture) > 0:
            gesture = np.asarray(simu.gesture)

            if gesture.shape[2] >= MIN_GESTURE_LEN:
                print("Recognizing Gesture...")
                simu.preprocess()
                print(labels[clf.predict(np.asarray(simu.gesture))[0]])
                sleep(1.5)
                print("Waiting for gesture...")
            else:
                print("Gesture too short.")

            simu.gesture = []
