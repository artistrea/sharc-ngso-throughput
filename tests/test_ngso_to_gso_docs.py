"""
Keep the ngso2gso parameter docs in sync with the code.

scenarios/ngso2gso_docs/ has example parameter files and a JSON schema
(English and pt-BR) describing every ngso2gso parameter. These tests fail
when a parameter is added, removed or renamed in the code without updating
the docs, or when the docs drift from each other.
"""
import copy
import dataclasses
import json
import unittest
import warnings
from pathlib import Path

import yaml

from sharc.parameters.parameters_ngso_to_gso import (
    ParametersGSO, ParametersGSOEarthStation, ParametersNGSO,
    ParametersNGSO2GSO, ParametersNGSOTxModel, SELECTION_STRATEGIES,
)
from sharc.parameters.parameters_orbit import ParametersOrbit

try:
    import jsonschema
except ImportError:
    jsonschema = None

DOCS_DIR = Path(__file__).resolve().parent.parent / "scenarios/ngso2gso_docs"
SCHEMA_FILES = ("ngso2gso_params.schema.json",
                "ngso2gso_params.schema.pt-BR.json")
EXAMPLES = sorted(DOCS_DIR.glob("exemplo_*.yaml"))

# schema $defs entry -> parameters class it documents
SCHEMA_DEF_TO_CLASS = {
    "ngso2gso": ParametersNGSO2GSO,
    "ngso_system": ParametersNGSO,
    "tx_model": ParametersNGSOTxModel,
    "orbit": ParametersOrbit,
    "gso_link": ParametersGSO,
    "earth_station": ParametersGSOEarthStation,
}

# ParametersBase fields that are not ngso2gso parameters: loader internals,
# and is_space_to_earth, which only the IMT simulator reads
INTERNAL_FIELDS = {
    "section_name", "nested_parameters_enabled", "is_space_to_earth",
}

# keys that hold human-readable text, allowed to differ between languages
TEXT_KEYS = {"description", "$comment", "title", "$id"}


def load_schema(name: str) -> dict:
    """Load one of the schema files."""
    return json.loads((DOCS_DIR / name).read_text())


def user_fields(cls) -> dict:
    """Map of parameter name -> default value for a parameters class."""
    fields = {}
    for f in dataclasses.fields(cls):
        if f.name in INTERNAL_FIELDS:
            continue
        if f.default is not dataclasses.MISSING:
            fields[f.name] = f.default
        else:
            fields[f.name] = dataclasses.MISSING  # default_factory
    if cls is ParametersNGSO2GSO:
        # class attribute, not a dataclass field, but read from the file
        fields["version"] = cls.version
    return fields


def strip_text(node):
    """Copy of a schema without its human-readable text."""
    if isinstance(node, dict):
        return {
            k: strip_text(v) for k, v in node.items() if k not in TEXT_KEYS
        }
    if isinstance(node, list):
        return [strip_text(v) for v in node]
    return copy.deepcopy(node)


class NGSO2GSODocsTest(unittest.TestCase):
    """The ngso2gso docs must describe the parameters the code accepts."""

    def test_docs_exist(self):
        """Sanity check: the docs are where these tests expect them."""
        self.assertEqual(len(EXAMPLES), 3)
        for name in SCHEMA_FILES:
            self.assertTrue((DOCS_DIR / name).is_file(), name)

    def test_examples_load_and_validate(self):
        """Every example parameter file is accepted by the simulator."""
        for example in EXAMPLES:
            with self.subTest(example=example.name), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                par = ParametersNGSO2GSO()
                par.load_parameters_from_file(example)
                par.validate("ngso2gso")

    def test_schema_lists_exactly_the_code_parameters(self):
        """Each schema object has one property per class parameter."""
        for name in SCHEMA_FILES:
            defs = load_schema(name)["$defs"]
            for def_name, cls in SCHEMA_DEF_TO_CLASS.items():
                with self.subTest(schema=name, definition=def_name):
                    self.assertEqual(
                        set(defs[def_name]["properties"]),
                        set(user_fields(cls)),
                        f"{name} $defs.{def_name} vs {cls.__name__}",
                    )

    def test_schema_defaults_match_code(self):
        """Defaults stated in the schema are the ones the code uses."""
        for name in SCHEMA_FILES:
            defs = load_schema(name)["$defs"]
            for def_name, cls in SCHEMA_DEF_TO_CLASS.items():
                fields = user_fields(cls)
                for prop, spec in defs[def_name]["properties"].items():
                    if "default" not in spec:
                        continue
                    with self.subTest(schema=name, parameter=prop):
                        self.assertEqual(
                            spec["default"], fields[prop],
                            f"{name} $defs.{def_name}.{prop}",
                        )

    def test_schema_selection_strategies_match_code(self):
        """The selection strategy enum lists the strategies the code has."""
        for name in SCHEMA_FILES:
            enum = load_schema(name)["$defs"]["selection_strategy"]["enum"]
            self.assertEqual(sorted(enum), sorted(SELECTION_STRATEGIES))

    def test_translations_only_differ_in_text(self):
        """The pt-BR schema is the English one with translated text."""
        en, pt = (strip_text(load_schema(name)) for name in SCHEMA_FILES)
        self.assertEqual(en, pt)

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_examples_match_schema(self):
        """Every example parameter file is valid against both schemas."""
        for name in SCHEMA_FILES:
            validator = jsonschema.Draft202012Validator(load_schema(name))
            for example in EXAMPLES:
                with self.subTest(schema=name, example=example.name):
                    errors = [
                        f"{list(e.absolute_path)}: {e.message}"
                        for e in validator.iter_errors(
                            yaml.safe_load(example.read_text())
                        )
                    ]
                    self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
