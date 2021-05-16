STATE_FINISH = 1
STATE_CONTINUE = 2
STATE_REPAIR = 3
CHANGE_BUBBLE = 4


class Manager:
    def __init__(self, initial_bubble):
        self.bubbles = []
        self.cur_bubble = initial_bubble

    def apply(self, intent, entity, text, user_belief, recommender, explainer):
        output = self.current_bubble.apply_bubble(
            intent, entity, text, user_belief, recommender, explainer
        )

        if output["state"] == STATE_CONTINUE:
            return output["utter"]

        elif output["change_bubble"] == True:

            self.cur_bubble = self.cur_bubble.get_next_bubble()
            start_utter = self.cur_bubble.start_bubble(user_belief)
            output["utter"].append(start_utter)
            return output["utter"]
