from typing import List
from neureca.explainer import Explainer
from neureca.recommender import Recommender


class UserBelief:
    def __init__(
        self,
        recommender: Recommender,
        explainer: Explainer,
    ):

        self._attributes = dict()
        self._recommender = recommender
        self._explainer = explainer

        self.user_name: str
        self._item_history: List[str] = list()
        self._recent_recommended_item: str

    def update_attributes(self, attributes):
        for k, v in attributes.items():
            if k not in self._attributes:
                self._attributes[k] = list()
            self._attributes[k] += v

        # print(self._attributes)

    def add_item_history(self, item_history: List[str]):
        self._item_history += item_history

    def has_attribute(self, attr_name):
        return attr_name in self._attributes

    def set_no_preference(self, attr_name):
        self._attributes[attr_name] = list()

    def get_attribute_value(self, attr_name):
        return ", ".join(self._attributes[attr_name])

    def recommend(self):
        item_list = self._recommender.convert_name_to_item(self._item_history)
        rec = self._recommender.run(item_list)[0]
        self._recent_recommended_item = rec
        name_rec = self._recommender.convert_item_to_name(rec)
        return name_rec

    def recommend_with_explanation(self):
        item_list = self._recommender.convert_name_to_item(self._item_history)
        rec_list = self._recommender.run(item_list)
        flat_attr_list = [attr for sublist in self._attributes.values() for attr in sublist]
        rec_item, exp = self._explainer.explain_recommendation(
            attribute_list=flat_attr_list, item_list=rec_list
        )
        self._recent_recommended_item = rec_item
        name_rec = self._recommender.convert_item_to_name(rec_item)

        return name_rec, exp

    def answer_question(self, attributes, item_name=None):
        if item_name is None:
            item = self._recent_recommended_item

        if item is None:
            raise NotImplementedError()

        flat_attr_list = [attr for sublist in attributes.values() for attr in sublist]
        answer, is_success = self._explainer.answer_question(flat_attr_list, item)
        return answer, is_success
