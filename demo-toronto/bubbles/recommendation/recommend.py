import neureca.dialogue_manager.box as box


class Recommend(box.Box):
    def apply_box(self, intent, attributes, text, user_belief, explainer, recommender):
        if intent == "ASK_QUESTION" or intent == "PROVIDE_PREFERENCE":
            attr = None
            for k, v in attributes.items():
                # TODO should change logic
                if k == "place_name":
                    continue

                attr = explainer.convert_syn_to_attr(v[0])
                exp = explainer.answer_question([attr], user_belief["recommended_item"])
                output = {"utter": exp[k]["answer"], "state": self.state}
        else:
            item_list = recommender.convert_name_to_item(user_belief["place_name"])
            recs = recommender.run(item_list)
            attr_list = [user_belief["food_type"], user_belief["location"], user_belief["occasion"]]
            exp = explainer.explain_recommendation(attr_list, recs)
            name_recs = recommender.convert_item_to_name([exp["rec"]])[0]
            uttr = ["I recommend {}".format(name_recs)] + exp["exp"]
            print(uttr)
            output = {"utter": uttr, "state": self.state}
            user_belief["recommended_item"] = exp["rec"]

        return output

    def _action_start(self, user_belief):
        utter = ["Ok, I'll give you a recommendation..."]

        return utter