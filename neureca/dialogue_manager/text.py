from typing import Dict


class Text:
    def __init__(self, intent: str, text: str, attributes: Dict[str, str]):
        self.intent = intent
        self.text = text
        self.attributes = attributes

    def check_attribute_mentioned(self, attribute):
        pass
