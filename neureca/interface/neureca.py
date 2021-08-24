import json
import re
from pathlib import Path
from typing import Optional
from flask import Flask, render_template
from neureca import NLU, Recommender, Explainer, Manager, UserBelief


class Neureca:
    def __init__(
        self,
        nlu: NLU,
        recommender: Recommender,
        explainer: Explainer,
        dialogue_manager: Manager,
        path: Optional[Path] = None,
    ):
        if path is None:
            path = Path.cwd()

        nlu.load_model(path)
        recommender.load_model(path)
        explainer.set(path)

        self.nlu = nlu
        self.user_belief = UserBelief(recommender, explainer)
        self.dialogue_manager = dialogue_manager

    def run(self, env: str = "cli"):

        name = input("Enter the name of user: ")
        if len(name) == 0:
            name = "USER"
        _item_history = input('Enter the item history of user. e.g. "item1", "item2": ')

        item_history = list()
        for item in re.findall(r'".+?"|[\w-]+', _item_history):
            item_history.append(item[1:-1])

        self.user_belief.user_name = name
        self.user_belief.add_item_history(item_history)

        if env == "cli":
            self._run_cli()
        elif env == "web":
            self._run_web()
        else:
            raise ValueError("The value of 'env' must be either 'cli' or 'web'")

    def _run_cli(self):
        print("\n -----CONVERSATION START-----\n")
        uttr = self.dialogue_manager.start_manager(self.user_belief)
        uttr = " ".join(uttr)
        print(f"[NEURECA]: {uttr}")

        while True:
            user_input = input(f"[{self.user_belief.user_name}]: ")
            text_output = self.nlu.run(user_input)

            uttr = self.dialogue_manager.apply(text_output, self.user_belief)
            uttr = " ".join(uttr)
            print(f"[NEURECA]: {uttr}")

    def _run_web(self):
        app = Flask(__name__)
        self._build(app)
        app.template_folder = str(Path(__file__).resolve().parents[0] / "templates/")
        app.static_folder = str(Path(__file__).resolve().parents[0] / "static/")
        app.run(port=8080, host="0.0.0.0")

    def _build(self, app):
        @app.route("/request_chat/<text>", methods=["GET"])
        def get_request(text):

            text_output = self.nlu.run(text)
            output = self.dialogue_manager.apply(text_output, self.user_belief)
            output = {"text": output}
            return json.dumps(output)

        @app.route("/start_chat", methods=["GET"])
        def start_chat():
            output = self.dialogue_manager.start_manager(user_belief=self.user_belief)
            output = {"text": output}

            return json.dumps(output)

        @app.route("/")
        def index():
            return render_template("index.html")
