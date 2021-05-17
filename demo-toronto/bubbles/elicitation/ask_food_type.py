import neureca.dialogue_manager.box as box


class AskFoodType(box.Box):
    def apply_box(self, intent, attributes, text, user_belief, **unused):
        if intent == "PROVIDE_PREFERENCE":
            user_belief.update(attributes)

            if user_belief["food_type"] is not None:
                utter = self._action_finished(user_belief)
                self.prev_action = None
                self.state = box.STATE_FINISH

            else:
                utter = self._action_ask_again()
                self.prev_action = self._action_ask_again
                self.state = box.STATE_CONTINUE
        else:
            utter = self._action_fallback()
            self.state = box.STATE_CONTINUE

        output = {"utter": utter, "state": self.state}

    def _action_fallback(self):
        utter = ["what the heck are you talking about?"]
        return utter

    def _action_start(self, **unused):
        utter = ["Do you have food type preference"]
        return utter

    def _action_finished(self, user_belief):
        utter = ["{}, nice! ".format(user_belief["food_type"])]
        return utter

    def _action_ask_again(self):
        utter = ["could you elaborate more?"]
        return utter