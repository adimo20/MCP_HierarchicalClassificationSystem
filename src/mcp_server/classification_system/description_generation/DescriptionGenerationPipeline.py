from classification_system.description_generation.label_generation.label_augmentation import LabelDescriptionGenerator
from classification_system.description_generation.dataloaders.CoicopDataloader import CoicopDataLoader
from classification_system.description_generation.dataloaders.KlassServerDataloader import XMLDataLoader
from classification_system.description_generation.dataloaders.NaceDataloader import load_nace

from classification_system.classification_system import ClassificationSystem, Code
from typing import Literal
import argparse
import os

class DescriptionGenerationPipeline:
    def __init__(
        self,
        classification_name:Literal["SEA", "COICOP", "WZ", "KlassServer", "NACE"],
        path_classification_data:str,
        api_key:str,
        api_base:str,
        model_name:str
    )->None:
        self.classification_name = classification_name
        self.api_key=api_key
        self.api_base=api_base
        self.model_name = model_name
        self.path_classification_data = path_classification_data
        self.classification_data:dict = self._load_classification_system() 
        self.classification_system = self._initialise_classification_system()
        self.description_generator = self._initialise_description_generator()
        
    def _load_classification_system(self):
        if self.classification_name in ["SEA", "WZ", "KlassServer"]:
            return XMLDataLoader(self.path_classification_data).load_dataset()
        elif self.classification_name == "COICOP":
            return CoicopDataLoader(self.path_classification_data).load_dataset()
        elif self.classification_name == "NACE":
            return load_nace(self.path_classification_data)
        else:
            raise Exception("Classification Type not supportet!")

    def _initialise_classification_system(self)->ClassificationSystem:
        codes = [Code().from_dict(c) for c in self.classification_data]
        classification_system = ClassificationSystem(codes)
        for c in classification_system.codes:
            c.level = len(c.code)-1
        
        return classification_system

    def _initialise_description_generator(self)->LabelDescriptionGenerator:
        return LabelDescriptionGenerator(
            classification=self.classification_system,
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name
        )

    def generate_descriptions(self, max_depth:int, output_path:str)->None:
        self.description_generator.generate_descriptions(
            max_depth=max_depth,
            output_path=output_path
        )
        
def main():
    parser = argparse.ArgumentParser(description="Run Description Generation Pipeline")

    # Required arguments
    parser.add_argument("--classification-name", required=True)
    parser.add_argument("--path-classification-data", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    # Initialize and run
    generator = DescriptionGenerationPipeline(
        classification_name=args.classification_name,
        path_classification_data=args.path_classification_data,
        api_key=args.api_key,
        api_base=args.api_base,
        model_name=args.model_name
    )

    generator.generate_descriptions(
        max_depth=args.max_depth,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()