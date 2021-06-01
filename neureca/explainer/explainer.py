from pathlib import Path
import pickle
import pandas as pd
from summarizers import Summarizers
from neureca.explainer.utils.parse_attribute import create_db

REVIEW_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "data" / "ratings.csv"
ATTRIBUTE_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "data" / "attribute.yaml"
ATTRIBUTE_PICKLE_PATH = (
    Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "attributes.pkl"
)
DB_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "db.csv"


class Explainer:
    def __init__(self):
        if not DB_PATH.exists():
            attributes = create_db(REVIEW_PATH, ATTRIBUTE_PATH, DB_PATH)
            with open(str(ATTRIBUTE_PICKLE_PATH), "wb") as f:
                pickle.dump(attributes, f)

        with open(str(ATTRIBUTE_PICKLE_PATH), "rb") as f:
            self.attributes = pickle.load(f)

        self.df = pd.read_csv(DB_PATH)
        self.summ = Summarizers()

    def convert_syn_to_attr(self, syn):
        for a in self.attributes:
            if a.check(syn):
                return a.name

    def _item_attr_analysis(self, item, attribute_list):
        if not isinstance(attribute_list, list):
            attribute_list = [attribute_list]

        df_item = self.df[self.df.item == item]
        print(df_item)
        sent_output = dict()

        for attr in attribute_list:
            df_item_attr = df_item[df_item.attr == attr]
            if len(df_item_attr) == 0:
                sent_output[attr] = None
            else:
                sent_output[attr] = df_item_attr["sentiment"].mean()
        return sent_output

    def explain_recommendation(self, attribute_list, item_list, topK=1):

        output = dict()
        for item in item_list:
            attr_sentiments = self._item_attr_analysis(item, attribute_list)
            print(list(attr_sentiments.values()))

            if None not in attr_sentiments.values():
                answers = self.answer_question(attribute_list, item)
                exp = []
                for k, v in answers.items():
                    exp.append(v["answer"])

                exp = " ".join(exp)
                print(exp)
                query = ", ".join(attribute_list)
                exp = [self.summ(exp, query=query)]

                return {"rec": item, "exp": exp}

            #    # exp = self._summarize(item, attribute_list)
            #    output[item] = exp

            if len(output) == topK:
                break

        return output

    def answer_question(self, attribute_list, item):
        answer_output = dict()

        attr_sent_dict = self._item_attr_analysis(item, attribute_list)
        print(attr_sent_dict)

        for attr in attr_sent_dict:
            answer_output[attr] = {"sentiment": None, "answer": None}

            if attr_sent_dict[attr] is not None:
                df_item_attr = self.df[self.df.item == item][self.df.attr == attr]
                review_lines = df_item_attr["review_line"].values
                print(review_lines)

                answer = " ".join(review_lines)
                answer = self.summ(answer, query=attr)
                answer_output[attr]["sentiment"] = attr_sent_dict[attr]
                answer_output[attr]["answer"] = answer

        return answer_output


if __name__ == "__main__":
    a = Explainer()
    z = a.explain_recommendation(
        item_list=["y0QzKWNVoXCbZpk6uhEgGA"],
        attribute_list=["italian", "date", "parking"],
    )
    print(z)
    print(a.convert_syn_to_attr("atmosphere"))
    print(a.convert_syn_to_attr("uber"))
