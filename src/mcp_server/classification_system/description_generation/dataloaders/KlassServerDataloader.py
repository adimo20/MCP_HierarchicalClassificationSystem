from pathlib import Path
import xml.etree.ElementTree as ET

class XMLDataLoader:
  """
  Can read all of the classification systems from the klass server.
  """
  def __init__(self, path) -> None:
    self.xml_file = Path(path)

  def get_detail_dict(self, item):

      """
      Constructs a dictionary for all the relevant sea categories with the given class names from the xml file.
      Parameters:
        item:xml.etree.ElementTree.Element
      Returns:
        dict
      """

      return {
          "code":item.get("id"),
          "description":item.find(".//Label[@qualifier='Usual']").find("LabelText").text,
          "level":item.get("idLevel"),
          "details":{
              "keywords":[
                  i.find(".//PropertyText[@type='Content']").text  if i.find(".//PropertyText[@type='Content']") is not None else ""   for i in item.findall(".//Property[@name='Keyword']")],
              "ExplanatoryNote":{
                  "exclusions":[ex.find(".//PropertyQualifier[@name='Exclusions']").find("PropertyText").text if ex.find(".//PropertyQualifier[@name='Exclusions']") is not None else "" for ex in item.findall(".//Property[@name='ExplanatoryNote']")],
                  "explicit_inclusion":[ex.find(".//PropertyQualifier[@name='CentralContent']").find("PropertyText").text if ex.find(".//PropertyQualifier[@name='CentralContent']") is not None else "" for ex in item.findall(".//Property[@name='ExplanatoryNote']")]
              },
              "context":item.find(".//Label[@qualifier='Context']").find("LabelText").text if item.find(".//Label[@qualifier='Context']") is not None else "",
          }
      }

  def parse_xml(self, path):
    """
    Parses the sea-documentation xml file and extracts all the relevant information.
    Parameters:
      path:str
    Returns:
      list
    """
    tree = ET.parse(path)
    root = tree.getroot()
    classification = root.find("Classification")
    labels = classification.findall("Item")
    all_labels = [self.get_detail_dict(l_) for l_ in labels]
    return all_labels

  def clean_dict(self, all_labels):

    """
    Cleans the dictionary of empty values.
    Parameters:
      all_labels:list
    Returns:
      list
    """

    for label in all_labels:
      for key in list(label["details"].keys()):
        if key == "ExplanatoryNote":
          for key2 in list(label["details"]["ExplanatoryNote"].keys()):
            if label["details"]["ExplanatoryNote"][key2] == []:
              del label["details"]["ExplanatoryNote"][key2]
        if label["details"][key] == [""] or label["details"][key] == "" or label["details"][key] == [] or label["details"][key] == {}:
          del label["details"][key]
      if label["details"] == {}:
        del label["details"]

    return all_labels

  def load_dataset(self):
    """
    Loads the sea documentation dataset.
    Parameters:
      None
    Returns:
      list
    """
    all_labels = self.parse_xml(self.xml_file)
    all_labels = self.clean_dict(all_labels)
    return all_labels

