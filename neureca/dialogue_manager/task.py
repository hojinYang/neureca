from neureca.dialogue_manager.user_belief import UserBelief
from neureca.nlu.nlu import NLUOutput


TASK_COMPLETED = 1
TASK_CONTINUE = 2

# STATE_REPAIR = 3


class Task:
    def __init__(self):
        self._state = None

    def get_name(self):
        return type(self).__name__

    def start_task(self, user_belief: UserBelief) -> str:
        utter = self.start(user_belief=user_belief)
        self._state = TASK_CONTINUE

        return utter

    def apply(self, text: NLUOutput, user_belief: UserBelief) -> str:
        raise NotImplementedError

    def start(self, user_belief: UserBelief) -> str:
        raise NotImplementedError

    def complete_task(self):
        self._state = TASK_COMPLETED

    def is_task_completed(self):
        return self._state == TASK_COMPLETED
