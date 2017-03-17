LEARNING_RATE = 0.1

"""
This class stored information to be carried between frames in video
"""


class FrameInfoStore:
    def __init__(self):
        self.heat_map = None
        self.svc = None
        self.scaler = None

    def initialized(self):
        return self.heat_map is None

    def clear(self):
        self.heat_map = None

    def update(self, new_heat_map):
        if self.heat_map is None:
            self.heat_map = new_heat_map
        else:
            self.heat_map = new_heat_map * LEARNING_RATE + self.heat_map * (1 - LEARNING_RATE)
