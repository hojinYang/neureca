from typing import List
from neureca.dialogue_manager import Stage, UserBelief
from neureca.nlu.nlu import NLUOutput


class Manager:
    def __init__(self, initial_stage: Stage):

        self.current_stage = initial_stage

    def start_manager(self, user_belief: UserBelief) -> List[str]:
        utter = self.current_stage.start_stage(user_belief=user_belief)
        return utter

    def apply(self, text: NLUOutput, user_belief: UserBelief) -> List[str]:
        utter = self.current_stage.apply(
            text=text,
            user_belief=user_belief,
        )

        if self.current_stage.is_stage_completed():
            self.current_stage = self.current_stage.next_stage
            start_utter = self.current_stage.start_stage(user_belief=user_belief)
            utter += start_utter

        return utter
