from dataclasses import dataclass, field, fields, asdict
import re
import json

@dataclass
class Code:
    code: str = field(default_factory=str)
    description: str = field(default_factory=str)
    level: str = field(default_factory=str)
    detailled_description: str = field(default_factory=str)
    details: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        valid_fields: set[str] = {f.name for f in fields(cls)}
        cleaned_data: dict = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**cleaned_data)

    def to_json(self, indent: int = None) -> str:
        """Converts the dataclass instance to a JSON string."""
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)
    
    def to_dict(self) -> dict:
        """Converts the dataclass instance to a JSON string."""
        return asdict(self)
  
  

@dataclass
class ClassificationSystem:
    codes: list[Code]
    _lookup: dict[str, Code] = field(init=False, repr=False)
    _tree: dict[str, list[Code]] = field(init=False, repr=False)


    def __post_init__(self):
        """
        Generating a lookup dict containing the code strings and Codes
        """
        self._lookup:dict = {self._preprocess_label(c.code):c for c in self.codes}
        for code in self._lookup.keys():
            self._lookup[code].code = self._preprocess_label(code=code)

        self._tree:dict = {}

        for c in self.codes:
            preprocessed_code: str = self._preprocess_label(code=c.code)
            parent: str = preprocessed_code[:-1]
            if parent not in self._tree.keys():
                self._tree.update(
                    {
                        parent:[code for code in self.codes if self._is_child(parent, self._preprocess_label(code.code))]
                    }
                )
                                
    def add_code(
        self,
        code:Code
    )->None:
        """
        Adds a code to the standing classification system
        """
        code = self._preprocess_label(code)
        self.codes.append(code)
        self._lookup.update(
            {
                self._preprocess_label(code.code):code
            }
        )
        preprocessed_code: str = self._preprocess_label(code=code.code)
        parent: str = preprocessed_code[:-1]
        if parent not in self._tree.keys():
            children: list[Code] = [code_ for code_ in self.codes if self._is_child(parent, self._preprocess_label(code_.code))]
            if children != []:
                self._tree.update(
                    {
                        parent:children
                    }
                )

    def get_code(
        self,
        code:str
    )->Code:
        try:
            code_formatted = self._preprocess_label(code)
            return self._lookup[code_formatted]
        except Exception as e:
            print(f"Code: {code} and Code after preprocessing {code_formatted}")
            raise ValueError(f"Code: {code} and Code after preprocessing {code_formatted} is not inside the classification system! {e}")

    def _preprocess_label(
        self,
        code:str
    )->str:
        """
        Formats the label into one unique format. Labels only consists out of numbers and capital letters.
        Parameters:
            label:str
        Returns:
            str
        """
        return re.sub(r"[^0-9A-Za-z]", "", code)
    
    def _is_child(
        self,
        parent: str,
        potential_child: str
    ) -> bool:
        """Checks if the parent is related to the potential_child"""

        parent_formatted = self._preprocess_label(parent)
        potential_child_formatted = self._preprocess_label(potential_child)

        n_parent:int = len(parent)
        n_potential_child:int = len(potential_child)
        if n_parent+1 == n_potential_child and parent_formatted==potential_child_formatted[:n_parent]:
            return True
        return False

    def get_children(
        self,
        parent:str
    )->list[Code]:
        """
        Collects a list of child categories for a given parent.
        Parameters:
            parent (str): The code you want to explore the children of (e.g., '01' or '011').
        Returns:
            List of child categories: Code
        """
        parent_formatted: str = self._preprocess_label(parent)
        try: 
            children: list[Code] = self._tree[parent_formatted]
            return children
        except Exception as e:
            print(f"Code {parent} hat no children.")
            raise ValueError(f"Code {parent} hat no children. Exeption: {e}")

    def get_code_trace(
        self,
        code:str
    )-> list[tuple]:
        """
        Returns the trace you would go in the hierarchy to reach the given code in form of a list of tuples(code, description)
        `Code trace` means in this case e.g. in the coicop **01 - Food and non-alcoholic beverages** -> **011 - Food** and so on.
        Works accordingly for NACE and other hierarchical classification system that follow the logic, that 011 or 012 is the child of 01
        Parameters:
            code (str): The code you want to get the trace from 
        Returns:
            list of tuples(code, description)
        """

        processed_code: str = self._preprocess_label(code)
        trace: list[str] = [processed_code[:i] for i in range(2, len(processed_code)+1)]
        
        valid_trace_tuples = []
        for t in trace:
            try:
                # Attempt to fetch the code. If it doesn't exist, it skips to the except block.
                c = self.get_code(t)
                valid_trace_tuples.append((t, c.description))
            except (ValueError, KeyError):
                # Silently skip missing parent codes (like dummy '99' prefixes)
                continue
                
        return valid_trace_tuples

    