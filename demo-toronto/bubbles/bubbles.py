import neureca.dialogue_manager.bubble as bubble
from .greeting import Greeting


class GreetingBubble(bubble.Bubble):
    def __init__(self, box_list):
        super().__init__(box_list)

    def set_start_box(self, user_belief):
        return "Greeting"

    def get_next_box(self, user_belief):
        if user_belief["user_name"] is not None:
            return bubble.CHANGE_BUBBLE
        else:
            return "Greeting"


greeting_bubble = GreetingBubble([Greeting()])
