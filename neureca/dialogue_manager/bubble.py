import neureca.dialogue_manager.box as box

CHANGE_BUBBLE = 0


class Bubble:
    def __init__(self, box_list):
        self.out_bubble = None

        self.box_dict = {box.get_name(): box for box in box_list}
        self.cur_box = None

        self.num_visited = 0
        self.bubble_state = False
        self.next_bubble = None

    def start_bubble(self, user_belief):
        box_name = self.set_start_box(user_belief)
        self.cur_box = self.box_dict[box_name]
        output = self.cur_box.start_box(user_belief=user_belief)
        return output

    def set_start_box(self, user_belief):
        pass

    def get_next_box(self, user_belief) -> str:
        """
        - User should implement this part
        - implement box change logic based on user belief state
        """
        pass

    def set_next_bubble(self, bubble):
        self.next_bubble = bubble

    def get_next_bubble(self):
        return self.next_bubble

    def apply_bubble(self, intent, attributes, text, user_belief):
        """
        dealing with user uttereance
        """

        bubble_output = dict()
        bubble_output["change_bubble"] = False

        # Box is the minimal component handling user utterence
        # current box should not be None

        # run
        box_output = self.cur_box.apply_box(intent, attributes, text, user_belief)
        bubble_output["utter"] = box_output["utter"]

        # case 1: user's information is not enough to finish current box task
        if box_output["state"] == box.STATE_CONTINUE:
            return bubble_output

        # case 2: finish current box task
        elif box_output["state"] == box.STATE_FINISH:
            box_name = self.get_next_box(user_belief)

            if box_name == CHANGE_BUBBLE:
                bubble_output["change_bubble"] = True

            else:
                self.cur_box = self.box_dict[box_name]
                start_utter = self.cur_box.start_box(user_belief)
                box_name["utter"].append(start_utter)

        return bubble_output
