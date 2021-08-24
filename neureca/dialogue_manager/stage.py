from typing import List
from neureca.dialogue_manager.user_belief import UserBelief
from neureca.dialogue_manager import task
from neureca.nlu.nlu import NLUOutput

STAGE_CONTINUE = 0
STAGE_COMPLETED = 1


class Stage:
    def __init__(self, task_list: List[task.Task]):
        self.task_dict = {task.get_name(): task for task in task_list}
        self.current_task: task.Task = None
        self.next_stage: Stage = None
        self._state = None

    def get_start_task(self, user_belief: UserBelief) -> str:
        raise NotImplementedError

    def get_next_task(self, user_belief: UserBelief) -> str:
        """
        - User should implement this part
        - implement task change logic based on user belief state
        """
        raise NotImplementedError

    def start_stage(self, user_belief: UserBelief) -> List[str]:
        self._state = STAGE_CONTINUE
        task_name = self.get_start_task(user_belief)
        self.current_task = self.task_dict[task_name]
        utter = self.current_task.start_task(user_belief=user_belief)
        return [utter]

    def apply(self, text: NLUOutput, user_belief: UserBelief) -> List[str]:
        """
        dealing with user uttereance
        """

        stage_output = dict()
        stage_output["change_stage"] = False

        # Task is the minimal component handling user utterence
        # current task should not be None

        # run
        utter = [self.current_task.apply(text, user_belief)]

        # case 1:  current task is completed
        if self.current_task.is_task_completed():
            name = self.get_next_task(user_belief)

            # current stage is completed
            if name == "Complete":
                self._state = STAGE_COMPLETED

            else:
                self.current_task = self.task_dict[name]
                start_utter = self.current_task.start_task(user_belief)
                utter.append(start_utter)

        # case 2: user's information is not enough to finish current task

        return utter

    def is_stage_completed(self):
        return self._state == STAGE_COMPLETED