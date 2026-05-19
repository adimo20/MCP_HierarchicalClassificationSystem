import pytest
from classification_system import ClassificationSystem, Code


@pytest.fixture(params=["clean", "dots", "spaces"])
def system_variant(request):
    """
    Provides a hypothetical ClassificationSystem instance, automatically cycling 
    through all three formatting variants (clean, dots, spaces) across all tests.

    In some cases, based on the template for the hierarchical classification system you'll use the codes
    will be formatted either in a clean way without seperators between hierarchy levels, e.g. `01111 - Rice`
    Sometimes the different hierachy levels are seperated by points `01.1.1.1 - Rice` or in case of the
    german SEA `0111 1 - Rice` by spaces. Those differences in format will be normalised inside the classifcation
    system instance. Thus we need to test all variants.
    """
    if request.param == "clean":
        codes = [
            Code(code="01", description="Test Division", level="1", detailled_description="...", details="..."),
            Code(code="011", description="Test Subgroup", level="2", detailled_description="...", details="..."),
            Code(code="0111", description="Test Class", level="3", detailled_description="...", details="..."),
            Code(code="0112", description="Test Class", level="3", detailled_description="...", details="..."),
            Code(code="012", description="Test Subgroup", level="2", detailled_description="...", details="..."),
            Code(code="0121", description="Test Class", level="3", detailled_description="...", details="..."),
        ]
    elif request.param == "dots":
        codes = [
            Code(code="01", description="Test Division", level="1", detailled_description="...", details="..."),
            Code(code="01.1", description="Test Subgroup", level="2", detailled_description="...", details="..."),
            Code(code="01.11", description="Test Class", level="3", detailled_description="...", details="..."),
            Code(code="01.12", description="Test Class", level="3", detailled_description="...", details="..."),
            Code(code="01.2", description="Test Subgroup", level="2", detailled_description="...", details="..."),
            Code(code="01.21", description="Test Class", level="3", detailled_description="...", details="..."),
        ]
    elif request.param == "spaces":
        codes = [
            Code(code="01", description="Test Division", level="1", detailled_description="...", details="..."),
            Code(code="01 1", description="Test Subgroup", level="2", detailled_description="...", details="..."),
            Code(code="01 1 1", description="Test Class", level="3", detailled_description="...", details="..."),
            Code(code="01 1 2", description="Test Class", level="3", detailled_description="...", details="..."),
            Code(code="01 2", description="Test Subgroup", level="2", detailled_description="...", details="..."),
            Code(code="01 2 1", description="Test Class", level="3", detailled_description="...", details="..."),
        ]
    return ClassificationSystem(codes=codes)

@pytest.mark.parametrize(
    "input_code, formatted_output",
    [
        ("01111", "01111"),
        ("01 1 1 1", "01111"),
        ("01.1 1.1", "01111"),
        ("A01111", "A01111"),
        ("b01 1 1 1", "b01111"),
        ("A01.1.1.1", "A01111"),
    ]
)
def test_label_normalisation(system_variant, input_code, formatted_output):
    assert system_variant._preprocess_label(input_code) == formatted_output


@pytest.mark.parametrize(
    "input_code, expected_code",
    [
       ("01" ,"01"),
       ("011" , "011"),
       ("01.1" , "011"),
       ("011" , "011"),
       ("0111", "0111"),
       ("01.1.1", "0111"),
       ("01 1 1", "0111"),
       ("0112", "0112"),
       ("01.1.2", "0112"),
       ("01 1 2", "0112"),
       ("012", "012"),
       ("01.2", "012"),
       ("01 2", "012"),
       ("0121", "0121"),
       ("01.2.1", "0121"),
       ("01 2 1", "0121")
    ]
)
def test_get_code(system_variant, input_code, expected_code):
    """
    Verifies that each code, even though its not normalised will retrieve its normalised version across all code formatting variations.
    """
    retrieved_code = system_variant.get_code(input_code).code
    assert expected_code == retrieved_code
    assert retrieved_code is not None


@pytest.mark.parametrize(
    "parent, expected_children",
    [
        ("01", ["011", "012"]),
        ("011", ["0111", "0112"]),
        ("012", ["0121"]),

        ("01 1", ["0111", "0112"]),
        ("01 2", ["0121"]),

        ("01.1", ["0111", "0112"]),
        ("01.2", ["0121"]),
    ]
)
def test_parent_children_relationships(system_variant, parent, expected_children):
    """Verifies that every parent code retrieves the correct children codes"""
    children_objects = system_variant.get_children(parent=parent)
    child_codes = [c.code for c in children_objects]
    assert child_codes == expected_children


@pytest.mark.parametrize(
    "code, expected_trace",
    [
        ("01", [("01", "Test Division")]),
        ("011", [("01", "Test Division"), ("011", "Test Subgroup")]),
        ("0111", [("01", "Test Division"), ("011", "Test Subgroup"), ("0111", "Test Class")]),
        ("0112", [("01", "Test Division"), ("011", "Test Subgroup"), ("0112", "Test Class")]),
        ("012", [("01", "Test Division"), ("012", "Test Subgroup")]),
        ("0121", [("01", "Test Division"), ("012", "Test Subgroup"), ("0121", "Test Class")]),
        
        ("01 1", [("01", "Test Division"), ("011", "Test Subgroup")]),
        ("01 1 1", [("01", "Test Division"), ("011", "Test Subgroup"), ("0111", "Test Class")]),
        ("01 1 2", [("01", "Test Division"), ("011", "Test Subgroup"), ("0112", "Test Class")]),
        ("01 2", [("01", "Test Division"), ("012", "Test Subgroup")]),
        ("01 2 1", [("01", "Test Division"), ("012", "Test Subgroup"), ("0121", "Test Class")]),

        ("01.1", [("01", "Test Division"), ("011", "Test Subgroup")]),
        ("01.1 1", [("01", "Test Division"), ("011", "Test Subgroup"), ("0111", "Test Class")]),
        ("01.1 2", [("01", "Test Division"), ("011", "Test Subgroup"), ("0112", "Test Class")]),
        ("01.2", [("01", "Test Division"), ("012", "Test Subgroup")]),
        ("01.2.1", [("01", "Test Division"), ("012", "Test Subgroup"), ("0121", "Test Class")]),
    ]
)
def test_code_traces(system_variant, code, expected_trace):
    """Verifies that the correct code traces will be extracted"""
    assert system_variant.get_code_trace(code=code) == expected_trace