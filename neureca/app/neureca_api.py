from typing import List
import json
from flask import Flask
from neureca import NLU, Recommender, Explainer, Explainer, Manager


class NeurecaApi:
    def __init__(self, nlu, recommender, explainer, dialogue_manager, initial_user_belief):

        self.nlu = nlu
        self.recommender = recommender
        self.explainer = explainer
        self.dialogue_manager = dialogue_manager
        # TODO: Maybe dialogue_manager contains recommender and explainer as attribute?
        self.user_belief = initial_user_belief

        self.app = Flask(__name__)
        self._build()

    def _build(self):
        @self.app.route("/request_chat/<text>", methods=["GET"])
        def get_request(text):

            nlu_output = self.nlu.run(text)
            print(nlu_output)
            output = self.dialogue_manager.apply(
                intent=nlu_output["intent"],
                attributes=nlu_output["attributes"],
                text=text,
                user_belief=self.user_belief,
                recommender=self.recommender,
                explainer=self.explainer,
            )
            print(output)
            print(self.user_belief)
            output = {"text": output["utter"]}
            return json.dumps(output)

        @self.app.route("/start_chat", methods=["GET"])
        def start_chat():
            output = self.dialogue_manager.start_manager(user_belief=self.user_belief)
            output = {"text": output["utter"]}

            return json.dumps(output)
