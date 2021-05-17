import neureca.dialogue_manager.box as box


class AskEssentialInfo(box.Box):
    def apply_box(self, intent, attributes, text, user_belief, **unused):
        if intent == "ASK_PLACE_RECOMMENDATION":
            user_belief.update(attributes)

            if user_belief["location"] is None and user_belief["occasion"] is None:
                utter = self._action_ask_location_and_occasion(user_belief)
                self.prev_action = self._action_ask_location_and_occasion
                self.state = box.STATE_CONTINUE

            elif user_belief["location"] is None:
                utter = self._action_ask_location(user_belief)
                self.prev_action = self._action_ask_location
                self.state = box.STATE_CONTINUE

            elif user_belief["occasion"] is None:
                utter = self._action_ask_occasion(user_belief)
                self.prev_action = self._action_ask_occasion
                self.state = box.STATE_CONTINUE

            else:
                utter = self._action_finished(user_belief)
                self.prev_action = None
                self.state = box.STATE_FINISH

        else:
            utter = self._action_fallback()
            self.state = box.STATE_CONTINUE

        output = {"utter": utter, "state": self.state}

        return output

    def _action_ask_location_and_occasion(self, user_belief):
        utter = ["Ok so can you tell me a bit detail about the context?"]
        return utter

    def _action_ask_occasion(self, user_belief):
        utter = ["{}, nice! what about occasion?".format(user_belief["location"])]
        return utter

    def _action_ask_location(self, user_belief):
        utter = ["{}, nice! how about location?".format(user_belief["occasion"])]
        return utter

    def _action_finished(self, user_belief):
        utter = ["{}, {}, nice! ".format(user_belief["occasion"], user_belief["location"])]
        return utter

    def _action_fallback(self):
        utter = ["what the heck are you talking about?"]
        return utter

    def _action_start(self, user_belief):
        utter = []

        if user_belief["occasion"] is None and user_belief["location"] is None:
            utter.append("I'll give you spot-on restaurant recommendation you'll love.")

        elif user_belief["location"] is None:
            utter.append(
                "I'll give you spot-on restaurant recommendation for {}.".format(
                    user_belief["occasion"]
                )
            )
        elif user_belief["occasion"] is None:
            utter.append(
                "I'll give you spot-on restaurant recommendation in {} you'll love.".format(
                    user_belief["location"]
                )
            )

        return utter