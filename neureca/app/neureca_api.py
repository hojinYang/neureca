from typing import List
from flask import Flask
from neureca.nlu.nlu import NLU
from neureca.recommender.recommender import Recommender
from neureca.dialogue_manager.manager import Manager
import json


class NeurecaApi:
    def __init__(self, nlu, recommender, dialogue_manager, initial_user_belief):

        self.nlu = nlu
        self.recommender = recommender
        self.dialogue_manager = dialogue_manager
        self.user_belief = initial_user_belief

        self.app = Flask(__name__)

        # self.featurizer = featurizer

        self._build()

    def _build(self):
        @self.app.route("/request_chat/<text>", methods=["GET"])
        def get_request(text):

            nlu_output = self.nlu.run(text)
            output_texts = self.dialogue_manager.apply(
                nlu_output, self.user_belief, self.recommender
            )
            return json.dumps({output_texts})
