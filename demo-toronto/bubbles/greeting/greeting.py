import neureca.dialogue_manager.box as box


class Greeting(box.Box):
    def apply_box(self, text, user_belief, **unused):
        user_belief["user_name"] = text

        utter = self._action_update_user_name(user_belief)
        self.state = box.STATE_FINISH
        self.prev_action = None

        output = {"utter": utter, "state": self.state}

        return output

    def _action_start(self, **unused):
        utter = ["Neuerca💡 here!", "Please enter your name to start conversation."]

        return utter

    def _action_update_user_name(self, user_belief):
        uname = user_belief["user_name"]
        utter = ["Hello {}!".format(uname)]
        return utter
