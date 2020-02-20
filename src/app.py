import cv2

from brain import BrainCore

brain = BrainCore()

while True:
    brain.start_cycle()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
