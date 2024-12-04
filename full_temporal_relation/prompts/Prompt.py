from doctest import Example
from typing import Optional

class Prompt:
    pass


class MultiEvents(Prompt):
    def __init__(self, use_vague = True, use_few_shot: bool = False, provide_justification: bool = False):
        self.use_vague = use_vague
        self.use_few_shot = use_few_shot
        self.provide_justification = provide_justification

        self.system = f"""
        Task Overview:
        You are given a text, in which some verbs are uniquely marked by [EVENT#ID]event[/EVENT#ID] (e.g., [EVENT1]event1[/EVENT1], [EVENT2]event2[/EVENT2]).
        Your task is to say which of the verbs happened first in a chronological order.
        More specifically, you need to return for each pair of verbs, which is two sentence apart,
        a single label out of the listed potential labels:
        before - the first verb happened before the second.
        after - the first verb happened after the second.
        equal - both verbs happened together.
        {"vague - It is impossible to know based on the context provided" if self.use_vague else ""}

        All responses should be a valide compact json format.
        The output json format must contains the following keys:
        event1 - the first relation, should be the lower #ID
        event2 - teh second relation, should be the higher #ID
        relation - the relation classification from the listed potential labels above 
        """

        self.few_shot = """
        Example:
        #########
        ---
        Text for Analysis:
        NAIROBI, Kenya (AP) _
        Suspected bombs [EVENT1]exploded[/EVENT1] outside the U.S. embassies in the Kenyan and Tanzanian capitals Friday, [EVENT2]killing[/EVENT2] dozens of people, witnesses [EVENT3]said[/EVENT3].
        The American ambassador to Kenya was among hundreds [EVENT12]injured[/EVENT12], a local TV [EVENT4]said[/EVENT4].
        ``It was definitely a bomb,'' [EVENT5]said[/EVENT5] a U.S. Embassy official in Nairobi, who [EVENT6]refused[/EVENT6] to [EVENT7]identify[/EVENT7] himself. ``You can [EVENT8]see[/EVENT8] a huge crater behind the building, and a bomb [EVENT9]went[/EVENT9] off at the embassy in Tanzania at the same time,'' he [EVENT10]said[/EVENT10].
        ---
        the sample of correct labels are:
        {examples}
        #########
        """

        self.context = """
        ---
        Text for Analysis:
        {text}
        ---
        """

        self.instruction = """Respond only with valid JSON. Do not write an introduction or summary.
        the list of pairs are: [{"""

    def generate_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        full_prompt = self.system
        few_shot_examples = """
                [{"event1":"1", "event2":"2", "relation": "before"},
                {"event1":"3", "event2":"12", "relation": "after"}
                """ + ',{"event1":"4", "event2":"5", "relation": "vague"}]' if self.use_vague else "]" 

        if self.provide_justification:
            full_prompt = full_prompt + "\njustification - justify your classification from the provided text, explain in one short sentences"
            few_shot_examples = """
                [
                {"event1":"1", "event2":"2", "relation": "before", "justification": "since exploded happened before killing"},
                {"event1":"3", "event2":"12", "relation": "after", "justification": "said happened before injured"},
                """ \
                    + '{"event1":"4", "event2":"5", "relation": "vague", "justification": "a local TV said happened before a U.S. Embassy official in Nairobi said"}]' \
                    if self.use_vague else "]" 
        
        if self.use_few_shot:
            full_prompt = f"{full_prompt} \n{self.few_shot.format(examples=few_shot_examples)}"

        context = self.context.format(text=text)
        instruction = self.instruction.format(relations=relations)
        return f"""{full_prompt} \n{context} \n{instruction}"""
    
    def generate_dict_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        system_content = self.system
        few_shot_examples = """
                [
                {"event1":"1", "event2":"2", "relation": "before"},
                {"event1":"3", "event2":"12", "relation": "after"}
                """ + ',{"event1":"4", "event2":"5", "relation": "vague"}]' if self.use_vague else "]" 

        if self.provide_justification:
            system_content = system_content + "\njustification - justify your classification from the provided text, explain in one short sentences"
            few_shot_examples = """
                [
                {"event1":"1", "event2":"2", "relation": "before", "justification": "since exploded happened before killing"},
                {"event1":"3", "event2":"12", "relation": "after", "justification": "said happened before injured"},
                """ \
                    + '{"event1":"4", "event2":"5", "relation": "vague", "justification": "a local TV said happened before a U.S. Embassy official in Nairobi said"}]' \
                    if self.use_vague else "]"
        
        if self.use_few_shot:
            system_content = f"{system_content} \n{self.few_shot.format(examples=few_shot_examples)}"

        context = self.context.format(text=text)
        # instruction = self.instruction.format(relations=relations)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"},
        ]
    
class PairwisePrompt(Prompt):
    def __init__(self, use_vague = True, use_few_shot: bool = False):
        self.use_vague = use_vague
        self.use_few_shot = use_few_shot

        self.system = f"""
        Task Overview:
        You are given a text, in which some verbs are uniquely marked by [EVENT#ID]event[/EVENT#ID] (e.g., [EVENT1]event1[/EVENT1], [EVENT2]event2[/EVENT2]).
        Your task is to say which of the verbs happened first in a chronological order.
        More specifically, you need to return for each pair of verbs, which is two sentence apart,
        a single label out of the listed potential labels:
        before - the first verb happened before the second.
        after - the first verb happened after the second.
        equal - both verbs happened together.
        {"vague - It is impossible to know based on the context provided" if self.use_vague else ""}

        you should only provide one classification.
        """

        self.few_shot = """
        Examples:
        #########
        ---
        Text for Analysis:
        NAIROBI, Kenya (AP) _
        Suspected bombs [EVENT1]exploded[/EVENT1] outside the U.S. embassies in the Kenyan and Tanzanian capitals Friday, [EVENT2]killing[/EVENT2] dozens of people, witnesses said.
        --> before
        ---
        Text for Analysis:
        Suspected bombs exploded outside the U.S. embassies in the Kenyan and Tanzanian capitals Friday, killing dozens of people, witnesses [EVENT3]said[/EVENT3].
        The American ambassador to Kenya was among hundreds [EVENT12]injured[/EVENT12], a local TV said.
        --> after
        #########
        """

        self.context = """
        ---
        Text for Analysis:
        {text}
        ---
        """

        self.instruction = """in one word --> """

    def generate_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        full_prompt = self.system

        context = self.context.format(text=text)
        instruction = self.instruction.format(relations=relations)
        return f"""{full_prompt} \n{context} \n{instruction}"""
    
    def generate_dict_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        system_content = self.system
        if self.use_few_shot:
            system_content += f'\n{self.few_shot}'

        context = self.context.format(text=text)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"}
        ]
    