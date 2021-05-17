from neureca.dialogue_manager.box import Box
from neureca.dialogue_manager.box import STATE_CONTINUE, STATE_FINISH, STATE_REPAIR


class GetEssentialInfo(Box):
    def __init__(self):
        pass

    def apply_box(self, entity, intent, prev_utter_type, text, user_belief):
        if intent == "ASK_PLACE_RECOMMENDATION":
            user_belief.update(entity)

            if user_belief["location"] is None and user_belief["occation"] is None:
                utter = ["Which location and ocassion?"]
                state = STATE_CONTINUE

            elif user_belief["location"] is None:
                utter = ["Which location..?"]
                state = STATE_CONTINUE

            elif user_belief["occasion"] is None:
                utter = ["which occasion...?"]
                state = STATE_CONTINUE

            else:
                utter = ["ok nice..."]
                state = STATE_FINISH

        output = {"utter": utter, "state": state}

        return output