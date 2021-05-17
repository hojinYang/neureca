class Manager:
    def __init__(self, initial_bubble):
        self.bubbles = []
        self.cur_bubble = initial_bubble

    def start_manager(self, user_belief):
        return self.cur_bubble.start_bubble(user_belief)

    def apply(self, intent, entity, text, user_belief, recommender, explainer):
        output = self.cur_bubble.apply_bubble(
            intent, entity, text, user_belief, recommender, explainer
        )

        if output["change_bubble"] == True:

            self.cur_bubble = self.cur_bubble.get_next_bubble()
            start_utter = self.cur_bubble.start_bubble(user_belief)
            output["utter"].append(start_utter)
            return output["utter"]
        else:
            return output["utter"]
