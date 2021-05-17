from pathlib import Path
from flask import render_template

# from .dialogue_manager import bubbles
from neureca import NeurecaApi
from bubbles import greeting_bubble
from neureca import NLU, Recommender, Explainer, Manager

manager = Manager(initial_bubble=greeting_bubble)

nlu = NLU()
# recommender = Recommender()
# elicit_bubble = bubbles.ElicitBubble()
# explainer = Explainer()
# manager = Manager(initial_bubble={})
initial_user_belief = {"user_name": None, "food_type": None, "location": None, "occasion": None}

neureca = NeurecaApi(
    nlu=nlu,
    recommender={},
    explainer={},
    dialogue_manager=manager,
    initial_user_belief=initial_user_belief,
)


@neureca.app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":

    neureca.app.template_folder = str(Path(__file__).resolve().parents[0] / "templates/")
    neureca.app.static_folder = str(Path(__file__).resolve().parents[0] / "static/")
    neureca.app.run(port=8080, host="0.0.0.0")
