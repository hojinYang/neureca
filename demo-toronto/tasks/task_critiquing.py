from neureca.dialogue_manager import Task


class Critiquing(Task):
    def __init__(self):
        super().__init__()
        self._prev_text_memory = None

    def apply(self, text, user_belief):
        if text.intent == "ASK_QUESTION":
            response, has_answer = user_belief.answer_question(text.attributes)
            if not has_answer:
                response = "Sorry, I'm not really sure about that... But I can find another place for you with that feature."

            else:
                response += "I can recommend you another place with that feature, if you want."

        elif self._prev_text_memory is not None and self._prev_text_memory.intent == "ASK_QUESTION":
            if text.intent == "POSITIVE":
                user_belief.update_attributes(self._prev_text_memory.attributes)
                response = "Cool."
                self.complete_task()
            elif text.intent == "NEGATIVE":
                response = "I see. Do you have another question or preference?"
            else:
                # fallback
                response = "Couldn't get it. Do you have another question or preference?"

        elif text.intent == "PROVIDE_PREFERENCE" or "ASK_PLACE_RECOMMENDATION":
            user_belief.update_attributes(text.attributes)
            response = (
                "I see, thanks for the information! Let me refine the recommendation for you.."
            )
            self.complete_task()

        else:
            # Fallback
            response = "Couldn't get what you mean.. So do you have any question or preference?"

        self._prev_text_memory = text
        return response

    def start(self, user_belief):
        response = "Do you have any question or preference?"
        self._prev_text_memory = None
        return response
