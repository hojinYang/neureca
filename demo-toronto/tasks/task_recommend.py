from neureca.dialogue_manager import Task


class Recommend(Task):
    def apply(self, text, user_belief):
        rec, exp = user_belief.recommend_with_explanation()
        response = f"How about {rec}? {exp}"
        self.complete_task()
        return response

    def start(self, user_belief):
        response = "ok so now I'm ready!"
        return response
