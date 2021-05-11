STATE_FINISH = 1
STATE_CONTINUE = 2
STATE_REPAIR = 3
CHANGE_BUBBLE = 4


class Box:
    def __init__(self):
        return

    def get_name(self):
        return type(self).__name__

    def apply_box(self, entity, intent, prev_utter_type, text, user_belief):
        pass