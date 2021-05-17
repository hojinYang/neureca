import neureca.dialogue_manager.box as box


class Greeting(box.Box):
    def __init__(self):
        super().__init__()

    def apply_box(self, intent, attributes, text, user_belief):
        user_belief["name"] = text
        utter = self._action_get_user_name(user_belief)

        self.state = box.STATE_FINISH
        self.prev_action = None

        output = {"utter": utter, "state": self.state}

        return output

    def start_box(self, **unused):
        utter = self._action_start()
        self.state = box.STATE_START
        self.prev_action = self._action_start

        output = {"utter": utter}

        return output

    def _action_start(self):
        output = ["Neuerca💡 here!", "Please enter your name to start conversation."]

        return output

    def _action_get_user_name(self, user_belief):
        uname = user_belief["name"]
        output = [
            "Hello {}!".format(uname),
            "I'll give you spot-on restaurant recommendation you'll love.",
        ]
        return output


greeting = Greeting()
