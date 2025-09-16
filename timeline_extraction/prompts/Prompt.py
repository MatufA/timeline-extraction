from typing import List, Optional


class Prompt:
    pass


class MultiEvents(Prompt):
    def __init__(
        self,
        use_vague=True,
        use_few_shot: bool = False,
        provide_justification: bool = False,
    ):
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

        All responses should be valid and compact dot graph format.

        compact meaning:
        - do not mention transitive dependencies - if ei1 BEFORE ei2 and ei2 BEFORE ei3 don't write ei1 BEFORE ei3
        - do not mention symmetric relation - if ei1 BEFORE ei2 don't write ei2 AFTER ei1
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

        self.instruction = """Respond only with valid dot graph format with the approprite markers and attributes (like label). Do not write an introduction or summary.
        the graph:"""

    def generate_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        full_prompt = self.system
        few_shot_examples = """
                digraph {
                    "EVENT1" -> "EVENT2" [label="before"];
                    "EVENT1" -> "EVENT3" [label="before"];
                    "EVENT2" -> "EVENT3" [label="before"];
                    "EVENT1" -> "EVENT12" [label="before"];
                    "EVENT1" -> "EVENT4" [label="before"];
                    "EVENT2" -> "EVENT12" [label="before"];
                    "EVENT2" -> "EVENT4" [label="before"];
                    "EVENT3" -> "EVENT12" [label="after"];
                    "EVENT3" -> "EVENT4" [label="before"];
                    "EVENT12" -> "EVENT4" [label="before"];
                    "EVENT12" -> "EVENT5" [label="before"];
                    "EVENT4" -> "EVENT5" [label="vague"];
                    "EVENT5" -> "EVENT8" [label="vague"];
                    "EVENT5" -> "EVENT9" [label="after"];
                    "EVENT5" -> "EVENT10" [label="before"];
                    "EVENT8" -> "EVENT9" [label="after"];
                    "EVENT8" -> "EVENT10" [label="before"];
                    "EVENT9" -> "EVENT10" [label="before"];
                    }
                """
        # few_shot_examples += '"EVENT4" -> "EVENT5" [label="vague"];\n}' if self.use_vague else "}"

        if self.provide_justification:
            full_prompt = (
                full_prompt
                + "\njustification - justify your classification from the provided text, explain in one short sentences"
            )
            few_shot_examples = (
                """
                [
                {"event1":"1", "event2":"2", "relation": "before", "justification": "since exploded happened before killing"},
                {"event1":"3", "event2":"12", "relation": "after", "justification": "said happened before injured"},
                """
                + '{"event1":"4", "event2":"5", "relation": "vague", "justification": "a local TV said happened before a U.S. Embassy official in Nairobi said"}]'
                if self.use_vague
                else "]"
            )

        if self.use_few_shot:
            full_prompt = (
                f"{full_prompt} \n{self.few_shot.format(examples=few_shot_examples)}"
            )

        context = self.context.format(text=text)
        instruction = self.instruction.format(relations=relations)
        return f"""{full_prompt} \n{context} \n{instruction}"""

    def generate_dict_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        system_content = self.system
        few_shot_examples = """
                digraph {
                    "EVENT1" -> "EVENT2" [label="before"];
                    "EVENT3" -> "EVENT12" [label="after"];
                """
        few_shot_examples += (
            '"EVENT4" -> "EVENT5" [label="vague"];\n}' if self.use_vague else "}"
        )

        if self.provide_justification:
            system_content = (
                system_content
                + "\njustification - justify your classification from the provided text, explain in one short sentences"
            )
            few_shot_examples = (
                """
                [
                {"event1":"1", "event2":"2", "relation": "before", "justification": "since exploded happened before killing"},
                {"event1":"3", "event2":"12", "relation": "after", "justification": "said happened before injured"},
                """
                + '{"event1":"4", "event2":"5", "relation": "vague", "justification": "a local TV said happened before a U.S. Embassy official in Nairobi said"}]'
                if self.use_vague
                else "]"
            )

        if self.use_few_shot:
            system_content = (
                f"{system_content} \n{self.few_shot.format(examples=few_shot_examples)}"
            )

        context = self.context.format(text=text)
        # instruction = self.instruction.format(relations=relations)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"},
        ]


class PairwisePrompt(Prompt):
    def __init__(self, use_vague=True, use_few_shot: bool = False):
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
            system_content += f"\n{self.few_shot}"

        context = self.context.format(text=text)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"},
        ]


class BreakCycleEvents(Prompt):
    def __init__(self):
        # self.use_few_shot = use_few_shot

        self.system = """
        Task Overview:
        You are given a text, in which some events are uniquely marked by [EVENT#ID]event[/EVENT#ID] (e.g., [EVENT1]event1[/EVENT1], [EVENT2]event2[/EVENT2]),
        and a dot graph which represent chronological order with error, where some edges form cycles.
        Your task is to decide which pair to drop (by his unique_id), being concise and removing the minimum number of edges.
        Pay attention, I used classifier to choose the most fitted relation (label attribute in dot graph) 
        and score which represent the confidence of the classifier.

        relation meaning:
        before - the first verb happened before the second.
        after - the first verb happened after the second.
        equal - both events happen simultaneously
        vague - temporal order cannot be determined from the context
        """
        self.context = """
        ---
        Text for Analysis:
        {text}
        """

        self.instruction = (
            """Respond only with the unique_id list to drop (wrong label)"""
        )

    @staticmethod
    def _generate_dot_graph(relation):
        line_format = '"{eiid1}" -> "{eiid2}" [label="{relation}", score={score}, unique_id={idx}];'
        return (
            "digraph Chronology {"
            + "\n\t".join(
                line_format.format(
                    eiid1=rel["eiid1"],
                    eiid2=rel["eiid2"],
                    relation=rel["relation"],
                    score=rel["probs"],
                    idx=idx,
                )
                for idx, rel in enumerate(relation)
            )
            + "\n}"
        )

    def generate_prompt(self, text: str, relations: List[str] = None):
        assert text is not None, "text not provided"

        full_prompt = self.system

        context = (
            self.context.format(text=text)
            + "\n The inconsistent graph:\n"
            + self._generate_dot_graph(relations)
            + "\n---"
        )
        instruction = self.instruction.format(relations=relations)
        return f"""{full_prompt} \n{context} \n{instruction}"""

    def generate_dict_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        system_content = self.system

        context = (
            self.context.format(text=text)
            + "\n"
            + self._generate_dot_graph(relations)
            + "\n---"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"},
        ]


class NTBreakCycleEvents(Prompt):
    def __init__(self):

        self.system = """
        Task Overview:
        You are given a text, in which some events are uniquely marked by [EVENT#ID]event[/EVENT#ID] (e.g., [EVENT1]event1[/EVENT1], [EVENT2]event2[/EVENT2]),
        and a dot graph which represent chronological order with error, where some edges form cycles.
        Your task is to decide which pair to drop (by his unique_id), being concise and removing the minimum number of edges.
        Pay attention, I used classifier to choose the most fitted relation (label attribute in dot graph) 
        and score which represent the confidence of the classifier.

        relation meaning:
        before - the first verb happened before the second.
        after - the first verb happened after the second.
        simultaneous - both events happen simultaneously.
        vague - temporal order cannot be determined from the context.
        includes - one event or time span fully contains another within its duration.
        is_included - an event or time is completely contained within the duration of another.
        overlap - two events or times partially coincide, sharing some duration but not fully containing each other.
        """

        self.context = """
        ---
        Text for Analysis:
        {text}
        """

        self.instruction = (
            """Respond only with the unique_id list to drop (wrong label)"""
        )

    @staticmethod
    def _generate_dot_graph(relation):
        for rel in relation:
            if not rel["eiid1"].startswith("EVENT"):
                rel["eiid1"] = rel["eiid1"].replace("e", "EVENT")
                rel["eiid2"] = rel["eiid2"].replace("e", "EVENT")

        line_format = '"{eiid1}" -> "{eiid2}" [label="{relation}", score={score}, unique_id={idx}];'
        return (
            "digraph Chronology {"
            + "\n\t".join(
                line_format.format(
                    eiid1=rel["eiid1"],
                    eiid2=rel["eiid2"],
                    relation=rel["relation"],
                    score=rel["probs"],
                    idx=idx,
                )
                for idx, rel in enumerate(relation)
            )
            + "\n}"
        )

    def generate_prompt(self, text: str, relations: List[str] = None):
        assert text is not None, "text not provided"

        full_prompt = self.system

        context = (
            self.context.format(text=text)
            + "\n The inconsistent graph:\n"
            + self._generate_dot_graph(relations)
            + "\n---"
        )
        instruction = self.instruction.format(relations=relations)
        return f"""{full_prompt} \n{context} \n{instruction}"""

    def generate_dict_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        system_content = self.system
        context = (
            self.context.format(text=text)
            + "\n"
            + self._generate_dot_graph(relations)
            + "\n---"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"},
        ]


class BreakCycleEventsByRelabeling(Prompt):
    def __init__(self):
        self.system = """
        Task Overview:
        You are given a text, in which some events are uniquely marked by [EVENT#ID]event[/EVENT#ID] (e.g., [EVENT1]event1[/EVENT1], [EVENT2]event2[/EVENT2]),
        and a dot graph which represent chronological order with error, where some edges form cycles.
        Your task is to decide which pair to relabel with new relation, being concise and change the minimum number of edges and return only the modification edges.
        Pay attention, I used classifier to choose the most fitted relation (label attribute in dot graph) 
        and score which represent the confidence of the classifier.

        relation meaning:
        before - the first verb happened before the second.
        after - the first verb happened after the second.
        equal - both events happen simultaneously
        vague - temporal order cannot be determined from the context
        """
        self.context = """
        ---
        Text for Analysis:
        {text}
        """

        self.instruction = """Respond only with valid dot graph format with the approprite markers and attributes (like label) for the modified edges. Do not write an introduction or summary.
        the graph:"""

    @staticmethod
    def _generate_dot_graph(relation):
        line_format = '"{eiid1}" -> "{eiid2}" [label="{relation}", score={score}, unique_id={idx}];'
        return (
            "digraph Chronology {"
            + "\n\t".join(
                line_format.format(
                    eiid1=rel["eiid1"],
                    eiid2=rel["eiid2"],
                    relation=rel["relation"],
                    score=rel["probs"],
                    idx=idx,
                )
                for idx, rel in enumerate(relation)
            )
            + "\n}"
        )

    def generate_prompt(self, text: str, relations: List[str] = None):
        assert text is not None, "text not provided"

        full_prompt = self.system

        context = (
            self.context.format(text=text)
            + "\n The inconsistent graph:\n"
            + self._generate_dot_graph(relations)
            + "\n---"
        )
        instruction = self.instruction.format(relations=relations)
        return f"""{full_prompt} \n{context} \n{instruction}"""

    def generate_dict_prompt(self, text: str, relations: Optional[str] = None):
        assert text is not None, "text not provided"

        system_content = self.system

        context = (
            self.context.format(text=text)
            + "\n"
            + self._generate_dot_graph(relations)
            + "\n---"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n{self.instruction}"},
        ]
