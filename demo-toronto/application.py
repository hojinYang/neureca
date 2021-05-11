from flask import render_template
from .dialogue_manager import bubbles
from neureca.app.neureca_api import NeurecaApi
from neureca.dialogue_manager.manager import Manager
from neureca.nlu.nlu import NLU
from neureca.recommender.recommender import Recommender


nlu = NLU()
recommender = Recommender()
elicit_bubble = bubbles.ElicitBubble()
manager = Manager(initial_bubble=elicit_bubble)

neureca = NeurecaApi(nlu, recommender, manager, None)


@neureca.app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    neureca.app.template_folder = neureca.root_dir + "templates"
    neureca.app.static_folder = neureca.root_dir + "static"
    neureca.app.run(port=8080, host="0.0.0.0")
