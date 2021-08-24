from neureca.dialogue_manager import Task


class AskFoodType(Task):
    def apply(self, text, user_belief):

        if text.intent == "ASK_RECOMMENDATION" or "PROVIDE_PREFERENCE":
            user_belief.update_attributes(text.attributes)

            if user_belief.has_attribute("food_type"):
                food_type = user_belief.get_attribute_value("food_type")
                response = f"{food_type}... I see."
                self.complete_task()

            else:
                response = "ok ok.. but I haven't heard about food type?"

        elif text.intent == "NO_INTENT" and text.has_attribute("food_type"):
            user_belief.update_attributes(text.attributes)
            food_type = user_belief.get_attribute_value("food_type")
            response = f"{food_type}... I see."
            self.complete_task()

        elif text.intent == "NEGATIVE":
            user_belief.set_no_preference("food_type")
            response = "I see."
            self.complete_task()

        else:
            # Fallback
            response = "Couldn't get what you mean..So do you have prefered food?"
        return response

    def start(self, user_belief):
        response = "Ok, then which food types are you looking for?"
        return response
