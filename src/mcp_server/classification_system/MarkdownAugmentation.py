from classification_system.classification_system import Code, ClassificationSystem
import json

class MarkdownExample:

    """
    Markdown Example is the base class for the augmentation of the Codes.
    Meaning that it defines how we want to present the Codes, when an agent want to either get details about a certain code or queries
    a database and receives back a relevant code including examples. In case you want to adapt this repo or change it to the needs of a certainly
    different classification system, adjust this class to change how you want to present the codes to an agent. 

    MarkdownExample generates a clearly structured and meaningful markdown representation of a certain code. Markdown is used
    instead of just returning the codes in form of the json string from the classification system, because we can encode a structure
    into it that is more meaningful to an agent that plain json. 
    """

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
        """
        Creates a markdown element, that consists out of a header and a content element, 
        you can highlight the content and or steer the degree of of the header

        Args:
            header (str) - Name of the markdown Element you want to create, e.g. ## Detailled Description of the Code
            content (str) - The content body you want to show
            header_degree (int) - Degree of the the header, # or ## ...
            highlight_content (bool) - Indicates if you want to highlight the content body, default is false
        Returns:
            str: markdown formatted string
        """
        content_formatted:str = f"**{content}**" if highlight_content else content
        return f"{"#"*header_degree} {header}\n{content_formatted}\n"

    def generate_examples_part(
        self,
        examples:list[str]
    )->str:
        """
        Generates a string containing a markdown list of examples you want to show to the agents, e.g.:
        Args:
            examples (list[str]) - Examples you want to show to the model
        Returns:
            str
        Example:
            ["Käse", "Milch"] --> "## Beispiele \n* Käse \n* Milch\n"

        """
        return f"## Beispiele \n{"\n".join([f"* {ex}" for ex in examples])}"

    def format_traces_to_markdown(
        self,
        trace:list[tuple[str,str]]
    )->str:
        """
        Creates a formatted markdown list from the output of the ClassificationSystem().get_code_trace() function
        Args:
            trace (list[tuple[str,str]]) - output of the ClassificationSystem().get_code_trace(), in Form of e.g. [("01", "FOOD AND NON-ALCOHOLIC BEVERAGES"), ("011", "FOOD"), ...]
        Returns:
            str: markdown formatted string
        Example:
           output: "* `Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES** <br> \n* `Gruppe 011`: **FOOD** <br> \n ..."
        """
        i = 0
        formatted_traces = ""
        for code, title in trace:
            formatted_traces += f"`{self.taxanomy_of_labels[i]} {code}`: **{title}** <br> \n" 
            # Output looks like: `Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES** <br> \n and so on for the whole trace
            i+=1
        return formatted_traces

    def code_to_markdown(
        self,
        code:Code,
        trace:list[tuple],
        examples:list[str]|None=None,
        classification_name:str="SEA"
    )->str:

        """
        Create the whole code summary by joining the different markdown parts. Customizable to the needs of a classification system
        It's possible to cut out or add new components, as long as you return a string
        Args:        
            code (Code) - important the input used here is type Code, custom dataclass
            trace (list[tuple[str, str]])
            examples (list[str]|None) - List of examples you want to add to your code summary, can be left None, in this case no examples will be shown
            classification_name (str) - Name of the classification system you want to use, will be inserted into the markdown string
        Returns:
            str: comprehensive markdown formatted string that is understandable for agents
        """

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
        
        code_markdown_format:str = "\n".join(
            [
                self.header_plus_content(header=h, content=c, highlight_content=highlight)
                for h, c, highlight in conf
            ]
        )
         
        if examples is not None and examples != []:      
            code_markdown_format = code_markdown_format + self.generate_examples_part(
                examples=examples
            )

        return code_markdown_format 
        

class MarkdownReport:

    """
    Interface for generating detailled markdown summaries for the details of a set of codes, 
    can be used for showing the details of codes and examples, when an agent retrieves codes from
    a data source, like a chromaDB or wants to look up a certain code it needs specifications for
    """

    def __init__(self, path:str, classification_name:str):
        self.path = path
        self.classification_name = classification_name

    def __post_init__(self):
        """Opens the classification system json file and initialises the Classification System object."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        codes: list[Code] = [Code().from_dict(c) for c in data]
        self.classification = ClassificationSystem(codes=codes)

    def generate_markdown_summary(self, list_of_codes:list[str], examples_dict:dict[str,list[str]]=None):
        """
        Generates a comprehensive markdown report for a list of classification codes.

        This method retrieves the data for each provided code, constructs its 
        hierarchical trace (path) through the classification system, and formats 
        the information into a structured markdown string. Each entry includes 
        the category name, the code itself, a detailed description, and the 
        full classification path.

        Args:
            list_of_codes (list[str]): A list of code strings to be summarized 
                (e.g., ['01111', '01112']).
            examples_dict  (dict[str,list[str]]): containing codes as keys and a list containing the examples, by default None --> no examples will be shown

        Returns:
            str: A formatted markdown string containing summaries for all valid codes, 
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
                    code=code,
                    trace=trace,
                    examples=ex
                ) for code, trace, ex in zip(codes, trace_formatted, examples)
            ]
        )
