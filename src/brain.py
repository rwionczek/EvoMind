import cv2
import numpy

from senses import CameraSense


class BrainCore:
    senses = (CameraSense,)

    def start_cycle(self):
        signals = self.process_senses()

        self.visualize(signals)

    def process_senses(self):
        signals = numpy.array([])

        for sense_class in self.senses:
            sense = sense_class()

            sense.detect_signals()

            signals = sense.get_signals()

        return signals

    def visualize(self, signals):
        grayscale_view = cv2.cvtColor(signals, cv2.COLOR_BGR2GRAY)
        view_lines = cv2.Canny(grayscale_view, threshold1=100, threshold2=200)
        cv2.imshow('window', view_lines)
