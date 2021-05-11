class Manager:
    def __init__(self, initial_bubble):
        self.bubbles = []
        self.current_bubble = initial_bubble

    def apply(self, intent, entity, text, user_belief, recommender):
        output = self.current_bubble.apply_bubble(intent, entity, text, user_belief, recommender)

        if output["change_bubble"]:
            self.current_bubble = self.current_bubble.move_to_next_bubble()
            output_ = self.current_bubble.get_box_intro(user_belief)
            output.update(output_)

        return output
