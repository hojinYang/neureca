from neureca.dialogue_manager import Task


class Greeting(Task):
    def apply(self, text, user_belief):
        if text.intent == "GREETING" or text.intent == "POSITIVE":
            response = "OK cool, I'll give you spot on recommendations in Toronto!"
            self.complete_task()

        elif text.intent == "ASK_PLACE_RECOMMENDATION":
            user_belief.update_attributes(text.attributes)

            if user_belief.has_attribute("food_type"):
                food_type = user_belief.get_attribute_value("food_type")
                response = f"OK, I'll recommend {food_type} restaurant."
            elif user_belief.has_attribute("occation"):
                occasion = user_belief.get_attribute_value("occasion")
                response = f"OK, I'll recommend some cool restaurnts for {occasion}."
            else:
                response = "OK cool, I'll give you spot on recommendations!"

            self.complete_task()

        else:
            # fallback
            response = "Couldn't get what you mean"

        return response

    def start(self, user_belief):
        response = f"Neuerca💡 here! How's your day, {user_belief.user_name}?"
        return response
