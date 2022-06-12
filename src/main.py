from machine import Pin, I2C
from mpu9250 import MPU9250
from fusion import Fusion


class IMU:
    def __init__(
        self,
        i2c_id: int = 0,
        i2c_scl: Pin = Pin(1),
        i2c_sda: Pin = Pin(0),
        verbose: bool = False,
        motion_threshold: int = 1,
    ) -> None:
        self.fuse = Fusion()
        i2c = I2C(i2c_id, scl=i2c_scl, sda=i2c_sda)
        self.imu = MPU9250(i2c)

        self.motion_threshold = motion_threshold
        self.__verbose__ = verbose
        self.motion_count = 0
        self.stddev = []
        self.mean = []

        self.ax_prev = 0
        self.ay_prev = 0
        self.az_prev = 0

        self.heading_prev = 0
        self.pitch_prev = 0
        self.roll_prev = 0

        for _ in range(300):
            self.fuse.update(self.imu.accel.xyz, self.imu.gyro.xyz, self.imu.mag.xyz)

    @property
    def inmotion(self) -> bool:
        self.update()
        _inmotion = False
        # ax, ay, az = self.imu.accel.xyz
        heading, roll, pitch = self.fuse.heading, self.fuse.roll, self.fuse.pitch

        if (
            abs(heading - self.heading_prev) > self.motion_threshold
            or abs(pitch - self.pitch_prev) > self.motion_threshold
            or abs(roll - self.roll_prev) > self.motion_threshold
        ):
            self.motion_count += 1
            _inmotion = True

        else:
            self.motion_count = 0
            _inmotion = False

        if self.__verbose__:
            print("motion:", self.motion_count)

        self.heading_prev = heading
        self.pitch_prev = pitch
        self.roll_prev = roll

        if _inmotion:
            return True
        return False

    def update(self) -> tuple:
        return self.fuse.update(self.imu.accel.xyz, self.imu.gyro.xyz, self.imu.mag.xyz)


class Motion:
    def __init__(self, imu_: IMU) -> None:
        self.imu: IMU = imu_
        self.samples: list = []

    def update(self) -> None:
        ax, ay, az = self.imu.imu.accel.xyz
        mx, my, mz = self.imu.imu.mag.xyz
        curr = [self.imu.fuse.heading, self.imu.fuse.roll, self.imu.fuse.pitch, ax, ay, az, mx, my, mz]

        if self.imu.inmotion and self.imu.motion_count > 10:
            self.imu.update()
            self.samples = curr
        else:
            self.samples = []
            del curr, ax, ay, az, mx, my, mz


if __name__ == "__main__":
    imu = IMU(motion_threshold=1.2)
    motion = Motion(imu)

    count = 0
    while True:
        motion.update()
        count += 1
        if len(motion.samples) >= 1:
            for i, v in enumerate(motion.samples):
                print(str(v) + ("," if i < len(motion.samples) - 1 else ""), end="")
            print("," + str(count))
            # print()
        # print(
        #     "".join([str(v) + ("," if i < len(motion.samples) - 1 else "") for i, v in enumerate(s)])
        #     + ("," + str(count) + "\n" if len(motion.samples) > 1 else ""),
        #     end="",
        # )
