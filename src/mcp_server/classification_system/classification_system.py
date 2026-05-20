from dataclasses import dataclass, field, fields, asdict
import re
import json

@dataclass
class Code:
    """
    The code class works a the base datatype for storing, retrieving and working with 
    codes from the classification systems. More or less every hierarchical classification
    system in official statistics, like NACE or COICOP haven a extensive documentation available, 
    what always contains 
    * `code` - Code in Form of digitis, e.g. 01111
    * `description` - overall description of the content of the code, e.g. Cereals (ND)
    * `level` - the level of the code inside the systems hierarchy, e.g. the corresponding level for the code 01111 is 4.
    * `detailled_description` - A more detailled description of what should be classified inside of a certain category, can be found in e.g. the COICOP documentation under introductory_notes
    * `details` - here will be all other details stored, that come on top of all the previous informations, e.g. explicit exclusion, inclusion, etc. 
    """
    code: str = field(default_factory=str)
    description: str = field(default_factory=str)
    level: str = field(default_factory=str)
    detailled_description: str = field(default_factory=str)
    details: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Loads in a code form a dictionary and saves it as a Code object. 
        All keys that don't exist in the dict but that exist in the Code
        object will be left blank. It is imprtant that the codes inserted 
        here match the required datarypes defined, otherwise it will break here
        or deeper down the pipeline.
        """
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

    """
    Central class to organise and retrieve informations from the hierarchical classification system. It receives 
    a list of Code objects as inputs and makes them searchable through it's methods. Additionally it standardises the
    codes to a format, where only letters and numbers are used, indicating that all special characters and spaces will be
    removed because the do not have any semantic meaning inside of the classification systems. 

    **Semantic Meaning of Codes in classifcation systems**

    Hierarchical classification systems are usually structed into certain very generic and general top level division, 
    that devide themselfs into more and more specific sub- and sub-sub-groups. Example from the COICOP:
    
    ```markdown
    `01 FOOD AND NON-ALCOHOLIC BEVERAGES` - Level 1
    `011 FOOD` - Level 2
    `0111 Cereals and cereal products (ND)` - LEvel 3
    `01111 Cereals (ND)` - Level 4
    ```

    This indicates a hierarchical tree-like structure, where:
    * 011 is the `child` of 01 
    * 01 is the `parent` of 011
    And it is also meaning that the parent category contains all the elements that deeper down that hierarchy sharing the same root nodes.
    Usually the condition applied to identify a child-parent relation between code is if "XX" -> "XXY" = Shared root.


    """

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
        Args:
            code (Code) - important: not that this code is of Type `Code`

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
        """
        Looks up a code inside of the classification system and returns the details in form the custom datatype Code. 
        Applies preprocessing and code normalisation before lookup so we do not miss a code just due to not aligned 
        code formatting. 
        Args: 
            code (str) - e.g. 01111
        """
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
        This has to be done, because depending on what data source you'll use when working with a classification system
        codes that are equal are formatted different ways, e.g. `01.1.1.1` or `0111 1` --> will be mapped to `01111` so 
        we don't miss a code, when we look for it.
        Parameters:
            label (str)
        Returns:
            str
        """
        return re.sub(r"[^0-9A-Za-z]", "", code)
    
    def _is_child(
        self,
        parent: str,
        : str
    ) -> bool:
        """Checks if the parent is related to the potential_child, 
        is true when parent="XX" -> potential_child "XXY" = Shared root.
        Args: 
            parent (str) - Parent we want to check
             (str) - potential child we want to check for relation to parent code
        Returns:
            bool - True when potential_child is related parent, else False
        """

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
        Collects a list of all child categories for a given parent.
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
        `Code trace` means in this case e.g. in the coicop **01 - Food and non-alcoholic beverages** -> **011 - Food** -> **0111 Cereals and cereal products (ND)** and so on.
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
                c = self.get_code(t)
                valid_trace_tuples.append((t, c.description))
            except (ValueError, KeyError):
                # In case a certain code is not inside the classification system/or a trace cannot be identified, because of missing parent elements
                # the codes that are not inside the system will be skipped silently without breaking the flow.            
                continue
                
        return valid_trace_tuples

    