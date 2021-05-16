STATE_FINISH = 1
STATE_CONTINUE = 2
STATE_REPAIR = 3
CHANGE_BUBBLE = 4


class Box:
    def __init__(self):
        self.actions = dict()
        self.prev_action = None
        self.box_state = None

    def get_name(self):
        return type(self).__name__

    def apply_box(self, intent, attributes, text, user_belief):
        pass

    def start_box(self, user_belief):
        pass