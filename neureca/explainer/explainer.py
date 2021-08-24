from pathlib import Path
import pickle
import pandas as pd
from summarizers import Summarizers
from neureca.explainer.utils.parse_attribute import save_db_and_attr_list


class Explainer:
    def __init__(self):
        return

    def set(self, path: Path):
        review_path = path / "data" / "ratings.csv"
        attribute_path = path / "data" / "attribute.yaml"
        attribute_pickle_path = path / "preprocessed" / "exp-attribute.pkl"
        db_path = path / "preprocessed" / "db.csv"

        if not db_path.exists():
            save_db_and_attr_list(review_path, attribute_path, db_path, attribute_pickle_path)

        with open(str(attribute_pickle_path), "rb") as f:
            self.attributes = pickle.load(f)

        self.df = pd.read_csv(db_path)
        self.summ = Summarizers()

    def convert_syn_to_attr(self, syn_lst):
        # FIXME revise this

        if not isinstance(syn_lst, list):
            syn_lst = [syn_lst]

        ret = list()
        for syn in syn_lst:
            for attr in self.attributes:
                if attr.check(syn):
                    ret.append(attr.name)
        return ret

    def explain_recommendation(self, attribute_list, item_list, topK=1):
        attribute_list = self.convert_syn_to_attr(attribute_list)

        for item in item_list:
            flag = True
            for attr in attribute_list:
                if not self._does_item_has_attribute(item, attr):
                    flag = False
                    break

            if flag:
                answers = list()
                for attr in attribute_list:
                    df_item_attr = self.df[self.df.item == item][self.df.attr == attr]
                    review_lines = df_item_attr["review_line"].values
                    answer = " ".join(review_lines)
                    answers.append(self.summ(answer, query=attr))

                answer = " ".join(answers)
                query = ", ".join(attribute_list)
                exp = self.summ(answer, query=query)

                return item, exp

        return item_list[0], None

    def answer_question(self, attribute_list, item=None):

        attribute_list = self.convert_syn_to_attr(attribute_list)
        answers = list()

        for attr in attribute_list:
            df_item_attr = self.df[self.df.item == item][self.df.attr == attr]
            if len(df_item_attr) == 0:
                continue
            else:
                review_lines = df_item_attr["review_line"].values
                answer = " ".join(review_lines)
                answer = self.summ(answer, query=attr)
                answers.append(answer)

        is_success = len(answers) != 0
        answer_output = " ".join(answers)

        return answer_output, is_success

    def _does_item_has_attribute(self, item, attr):
        return len(self.df[self.df.item == item][self.df.attr == attr]) > 0
