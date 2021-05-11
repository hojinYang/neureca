from neureca.dialogue_manager.box import Box
from neureca.dialogue_manager.box import STATE_CONTINUE, STATE_FINISH, STATE_REPAIR


class GetAdditionallInfo(Box):
    def intro_utter(self, user_belief):
        return "additional info?"

    def apply_box(self, entity, intent, prev_utter_type, text, user_belief):
        if intent == "ASK_PLACE_RECOMMENDATION":
            # user_belief.update(entity)

            if user_belief["food_type"] is None:
                utter = ["What food?"]
                state = STATE_CONTINUE

        output = {"utter": utter, "state": state}

        return output
