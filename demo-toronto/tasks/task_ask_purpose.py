from neureca.dialogue_manager import Task


class AskPurpose(Task):
    def apply(self, text, user_belief):

        if text.intent == "ASK_PLACE_RECOMMENDATION" or "PROVIDE_PREFERENCE":
            user_belief.update_attributes(text.attributes)

            if user_belief.has_attribute("occasion"):
                occasion = user_belief.get_attribute_value("occasion")
                response = f"I see.. I'll recommend some cool restaurnts for {occasion}."
                self.complete_task()

            else:
                response = "ok ok.. but I haven't heard about your purpose, do you have any purpose?"

        elif text.intent == "NO_INTENT" and text.has_attribute("occasion"):
            user_belief.update_attributes(text.attributes)
            occasion = user_belief.get_attribute_value("occasion")
            response = f"I see.. I'll recommend some cool restaurnts for {occasion}."
            self.complete_task()

        elif text.intent == "NEGATIVE":
            user_belief.set_no_preference("occasion")
            response = "I see."
            self.complete_task()

        else:
            # Fallback
            response = "Couldn't get what you mean..So do you have specific purpose?"
        return response

    def start(self, user_belief):
        response = f"So do you have specific purpose, {user_belief.user_name}?"
        return response
