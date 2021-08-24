from pathlib import Path
from neureca import Neureca, NLU, Recommender, Explainer, Manager
from stages import greeting_stage
import warnings
 
warnings.filterwarnings("ignore")

# load the latest trained version of each model if version is not speicfied in arguament
nlu = NLU()  
recommender = Recommender()
explainer = Explainer()
dialogue_manager = Manager(initial_stage=greeting_stage)


neureca = Neureca(
    nlu=nlu,
    recommender=recommender,
    explainer=explainer,
    dialogue_manager=dialogue_manager,
    path=Path.cwd(),
)

if __name__ == '__main__':
    neureca.run(env="cli")  # env: cli or web
