STATE_START = 0
STATE_FINISH = 1
STATE_CONTINUE = 2
STATE_REPAIR = 3


class Box:
    def __init__(self):
        self.prev_action = None
        self.state = None

    def get_name(self):
        return type(self).__name__

    def start_box(self, user_belief):
        utter = self._action_start(user_belief=user_belief)
        self.state = STATE_START
        self.prev_action = self._action_start

        output = {"utter": utter}

        return output

    def apply_box(self, intent, attributes, text, user_belief, recommender, explainer):
        pass

    def _action_start(self, user_belief):
        pass