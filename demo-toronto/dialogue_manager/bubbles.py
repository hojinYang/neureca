from neureca.dialogue_manager.bubble import Bubble
from elicitation.get_essential_info import GetEssentialInfo
from elicitation.get_additional_info import GetAdditionallInfo


class ElicitBubble(Bubble):
    def __init__(self):
        super.__init__()

    def change_box(self, user_belief) -> str:
        if user_belief["location"] is None or user_belief["occasion"] is None:
            return "GetEssentialInfo"
        if user_belief["food_type"] is None:
            return "GetAdditionalInfo"
        return "ChangeBubble"
