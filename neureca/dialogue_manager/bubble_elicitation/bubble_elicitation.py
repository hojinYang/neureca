from neureca.dialogue_manager.bubble import BaseBubble


class BubbleElicitation(BaseBubble):
    def __init__(self, box_list):
        self.boxes = dict()
        self.cur_box = None
        self.next_bubble = None

        for box in box_list:
            self.boxes[box.name()] = box

    def update_cur_box(self, user_belief):
        if user_belief["location"] or user_belief["occasion"] is None:
            self.cur_box = self.boxes["ask_basic_preference"]

        if user_belief["food_type"] is None:
            self.cur_box = self.boxes["ask_food_type"]

        self.cur_box = self.next_bubble

    def apply_box(self, entity, text, user_belief):

        output = self.cur_box.apply(entity, text, user_belief)

        if output.state == "continue":
            return output

        if output.state == "repair":
            

        elif output.state == "done":
            self.update_cur_box(self, user_belief)

            if self.cur_box.initial_state is next_bubble:
                output.move_bubble = True
            else:
                output.utter.append(self.cur_box.intro)

        return output_state