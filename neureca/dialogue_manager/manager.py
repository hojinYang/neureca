class Manager:
    def __init__(self, initial_bubble):

        self.cur_bubble = initial_bubble

    def start_manager(self, user_belief):
        return self.cur_bubble.start_bubble(user_belief)

    def apply(self, intent, attributes, text, user_belief, recommender, explainer):
        output = self.cur_bubble.apply_bubble(
            intent=intent,
            attributes=attributes,
            text=text,
            user_belief=user_belief,
            recommender=recommender,
            explainer=explainer,
        )
        print(user_belief)

        if output["change_bubble"] == True:

            self.cur_bubble = self.cur_bubble.get_next_bubble()
            start_utter = self.cur_bubble.start_bubble(user_belief)
            output["utter"] += start_utter["utter"]

        return output
