from classification_system.classification_system import Code, ClassificationSystem
import json

class MarkdownExample:
    def __init__(
        self
    )->None:
        self.taxanomy_of_labels: list[str] = [
            "Abteilung",
            "Gruppe",
            "Klasse",
            "Unterklasse",
            "Kategorie",
            "Unterkategorie"
        ]
    
    def header_plus_content(
        self,
        header:str,
        content:str,
        header_degree:int=2,
        highlight_content:bool=False
    )->str:
        content_formatted:str = f"**{content}**" if highlight_content else content
        return f"{"#"*header_degree} {header}\n{content_formatted}\n"

    def generate_examples_part(
        self,
        examples:list[str]
    )->str:
        return f"## Beispiele \n{"\n".join([f"* {ex}" for ex in examples])}"

    def format_traces_to_markdown(
        self,
        trace:list[tuple[str,str]]
    )->str:
        i = 0
        formatted_traces = ""
        for code, title in trace:
            formatted_traces += f"`{self.taxanomy_of_labels[i]} {code}`: **{title}** <br> \n" 
            i+=1
        return formatted_traces

    def code_to_markdown(
        self,
        code:Code,
        trace:list[tuple],
        examples:list[str]|None=None,
        classification_name:str="SEA"
    )->str:

        conf:list[tuple[str, str, bool]] = [
            (
                "Name der Kategorie",
                code.description,
                False
                
            ),
            (
                f"{classification_name}-Code",
                code.code,
                True
            ),
            (
                "Detaillierte Beschreibung",
                code.detailled_description,
                False
            ),
            (
                f"Pfad der {classification_name}-Klassifikation",
                self.format_traces_to_markdown(trace=trace),
                False
            )
        ]
        
        code_markdown_format:str = "\n".join([self.header_plus_content(header=h, content=c, highlight_content=highlight) for h, c, highlight in conf])

        if examples is not None and examples != []:         
            code_markdown_format = code_markdown_format + self.generate_examples_part(
                examples=examples
            )

        return code_markdown_format 
        

class MarkdownReport:

    def __init__(self, path:str, classification_name:str):
        self.path = path
        self.classification_name = classification_name

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        codes: list[Code] = [Code().from_dict(c) for c in data]
        self.classification = ClassificationSystem(codes=codes)

    def generate_markdown_summary(self, list_of_codes:list[str], examples_dict:dict=None):
        """
        Generates a comprehensive Markdown report for a list of classification codes.

        This method retrieves the data for each provided code, constructs its 
        hierarchical trace (path) through the classification system, and formats 
        the information into a structured Markdown string. Each entry includes 
        the category name, the code itself, a detailed description, and the 
        full classification path.

        Args:
            list_of_codes (list[str]): A list of code strings to be summarized 
                (e.g., ['01111', '01112']).

        Returns:
            str: A formatted Markdown string containing summaries for all valid codes, 
                separated by horizontal rules (---).

        Example:
            generate_markdown_summary(['01111'], "SEA")
            '# Name der Kategorie: \n **Getreide** ...'
        """
        codes: list[Code] = [self.classification.get_code(code) for code in list_of_codes if code in self.classification._lookup]
    
        trace_formatted: list[str] = [
            self.classification.get_code_trace(code.code)
            for code in codes
        ]
        if examples_dict is not None:
            examples = [examples_dict.get(c.code, []) for c in codes]
        else: 
            examples = [[] for _ in range(len(codes))]
        
        return "\n---\n".join(
            [
                MarkdownExample().code_to_markdown(
                    code,
                    trace
                ) for code, trace, ex in zip(codes, trace_formatted, examples)
            ]
        )
