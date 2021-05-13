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

            # nlu_output = self.nlu.run(text)
            # output_texts = self.dialogue_manager.apply(
            #    nlu_output, self.user_belief, self.recommender, self.explainer
            # )
            output = {"text": "hihi"}
            return json.dumps(output)
