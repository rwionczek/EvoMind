from abc import abstractmethod

import mss
import numpy


class BaseSense:
    signals = None

    def get_signals(self):
        return self.signals

    @abstractmethod
    def detect_signals(self):
        pass


class CameraSense(BaseSense):
    def detect_signals(self):
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 960, "height": 540}

            self.signals = numpy.array(sct.grab(monitor))
