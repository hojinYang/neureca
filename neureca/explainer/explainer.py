from pathlib import Path
import pandas as pd
from summarizers import Summarizers
from neureca.explainer.utils.parse_attribute import create_db

REVIEW_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "data" / "ratings.csv"
ATTRIBUTE_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "data" / "attribute.yaml"

DB_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "db.csv"


class Explainer:
    def __init__(self):
        if not DB_PATH.exists():
            create_db(REVIEW_PATH, ATTRIBUTE_PATH, DB_PATH)

        self.df = pd.read_csv(DB_PATH)
        self.summ = Summarizers()

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
            # attr_sentiments = self._item_attr_analysis(item, attribute_list)

            # if None not in attr_sentiments:
            #    # exp = self._summarize(item, attribute_list)
            #    output[item] = exp

            if len(output) == topK:
                break

        return output

    def answer_question(self, attribute_list, item):
        answer_output = dict()

        attr_sent_dict = self._item_attr_analysis(item, attribute_list)

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
    z = a.answer_question(
        item="y0QzKWNVoXCbZpk6uhEgGA", attribute_list=["parking", "italian", "date"]
    )
    print(z)
