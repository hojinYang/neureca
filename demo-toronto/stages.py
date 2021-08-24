from neureca.dialogue_manager import Stage
from tasks import Greeting, AskPurpose, AskLocation, AskFoodType, Recommend, Critiquing


class GreetingStage(Stage):
    def get_start_task(self, user_belief):
        return "Greeting"

    def get_next_task(self, user_belief):
        return "Complete"


class ElicitationStage(Stage):
    def get_start_task(self, user_belief):
        return self.get_next_task(user_belief)

    def get_next_task(self, user_belief):
        if not user_belief.has_attribute("occasion"):
            return "AskPurpose"
        elif not user_belief.has_attribute("location"):
            return "AskLocation"
        elif not user_belief.has_attribute("food_type"):
            return "AskFoodType"
        return "Complete"


class RecommendationStage(Stage):
    def get_start_task(self, user_belief):
        self.critiquing_visited = False
        return "Recommend"

    def get_next_task(self, user_belief):
        if self.critiquing_visited is False:
            self.critiquing_visited = True
            return "Critiquing"
        else:
            self.critiquing_visited = False
            return "Complete"


greeting_stage = GreetingStage([Greeting()])
elicitation_stage = ElicitationStage([AskLocation(), AskPurpose(), AskFoodType()])
greeting_stage.next_stage = elicitation_stage
recommendation_stage = RecommendationStage([Recommend(), Critiquing()])
elicitation_stage.next_stage = recommendation_stage
recommendation_stage.next_stage = recommendation_stage
