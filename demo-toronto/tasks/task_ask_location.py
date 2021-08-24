from neureca.dialogue_manager import Task


class AskLocation(Task):
    def apply(self, text, user_belief):

        if text.intent == "ASK_PLACE_RECOMMENDATION" or text.intent == "PROVIDE_PREFERENCE":
            user_belief.update_attributes(text.attributes)

            if user_belief.has_attribute("location"):
                location = user_belief.get_attribute_value("location")
                response = f"In {location}... I see."
                self.complete_task()

            else:
                response = "ok ok.. but I haven't heard about location, do you have any preference?"

        elif text.intent == "NO_INTENT" and text.has_attribute("location"):
            user_belief.update_attributes(text.attributes)
            location = user_belief.get_attribute_value("location")
            response = f"In {location}... I see."
            self.complete_task()

        elif text.intent == "NEGATIVE":
            user_belief.set_no_preference("location")
            response = "I see."
            self.complete_task()

        else:
            # Fallback
            response = "Couldn't get what you mean..So do you have prefered location?"
        return response

    def start(self, user_belief):
        response = f"So {user_belief.user_name}, which location are you looking for?"
        return response
