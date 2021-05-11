STATE_FINISH = 1
STATE_CONTINUE = 2
STATE_REPAIR = 3
CHANGE_BUBBLE = 4


class Bubble:
    def __init__(self, box_list):
        self.out_bubble = None

        self.box_dict = {box.name: box for box in box_list}
        self.cur_box = None

    def change_box(self, user_belief) -> str:
        """
        - User should implement this part
        - implement box change logic based on user belief state
        """
        pass

    def update_cur_box(self, user_belief):
        box_name = self.change_box(user_belief)
        return self.box_dict[box_name]

    def apply_bubble(self, entity, text, prev_utter_type, user_belief):
        """
        dealing with user uttereance
        """
        bubble_output = dict()
        bubble_output["change_bubble"] = False
        bubble_output["prev_utter_type"] = None
        # Box is the minimal component handling user utterence
        # current box should not be None

        # run
        box_output = self.cur_box.apply_box(entity, text, prev_utter_type, user_belief)
        bubble_output["utter"] = box_output["utter"]

        # case 1: user's information is not enough to finish current box task
        if box_output["state"] == STATE_CONTINUE:
            bubble_output["prev_utter_type"] = box_output["prev_utter_type"]

        # case 2: finish current box task
        elif box_output["state"] == STATE_FINISH:
            box_name = self.change_box(user_belief)

            if self.cur_box == CHANGE_BUBBLE:
                bubble_output["change_bubble"] = True

                return bubble_output

            self.cur_box = self.box_dict[box_name]
            new_box_intro = self.cur_box.box_intro_utter(user_belief)
            bubble_output["utter"] += new_box_intro

        # case 3: user says some different stories that should be handled in different box
        elif box_output["state"] == STATE_REPAIR:
            # no need to give a hint to user that we are in this box, as user already know it.
            self.cur_box = self.box_dict[box_output["repair_box"]]

            repair_box_output = self.cur_box.apply_box(entity, text, prev_utter_type, user_belief)
            bubble_output["utter"] += repair_box_output["utter"]

            # case 3-1: user's information is not enough to finish current box task
            if repair_box_output["state"] == STATE_CONTINUE:
                bubble_output["prev_utter_type"] = repair_box_output["prev_utter_type"]

            # case 3-2: finish current box task
            elif repair_box_output["state"] == STATE_FINISH:
                box_name = self.change_box(user_belief)

                if self.cur_box == CHANGE_BUBBLE:
                    bubble_output["change_bubble"] = True

                    return bubble_output
                self.cur_box = self.box_dict[box_name]
                new_box_intro = self.cur_box.box_intro_utter(user_belief)
                bubble_output["utter"] += new_box_intro

        return bubble_output
