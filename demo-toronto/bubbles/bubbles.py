import neureca.dialogue_manager.bubble as bubble
from .greeting import Greeting
from .elicitation import AskEssentialInfo, AskFoodType
from .recommendation import Recommend


class GreetingBubble(bubble.Bubble):
    def set_start_box(self, user_belief):
        return "Greeting"

    def get_next_box(self, user_belief):

        if user_belief["user_name"] is not None:
            return bubble.CHANGE_BUBBLE
        else:
            return "Greeting"


class ElicitationBubble(bubble.Bubble):
    def set_start_box(self, user_belief):
        return self.get_next_box(user_belief)

    def get_next_box(self, user_belief):
        if user_belief["occasion"] is None or user_belief["location"] is None:
            return "AskEssentialInfo"
        if user_belief["food_type"] is None:
            return "AskFoodType"

        print("lets change bubble")
        return bubble.CHANGE_BUBBLE


class RecommendationBubble(bubble.Bubble):
    def set_start_box(self, user_belief):
        print("SDFSDKFSOGRMGRIMGRIM")
        return "Recommend"

    def get_next_box(self, user_belief):
        return "Recommend"


greeting_bubble = GreetingBubble([Greeting()])
elicitation_bubble = ElicitationBubble([AskEssentialInfo(), AskFoodType()])
recommendation_bubble = RecommendationBubble([Recommend()])
greeting_bubble.set_next_bubble(elicitation_bubble)
elicitation_bubble.set_next_bubble(recommendation_bubble)
