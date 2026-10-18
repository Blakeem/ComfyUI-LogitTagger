"""workflows/: each workflow's graph against the pack's node classes, and the README's workflow table against the folder.

The frontend's app.graph.serialize() wrote every file, and INPUT_TYPES() needs no comfy install, GPU or model.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote, unquote

import pytest

PACK_ROOT = Path(__file__).parent.parent.resolve()
WORKFLOW_DIR = PACK_ROOT / "workflows"
WORKFLOW_PATHS = sorted(WORKFLOW_DIR.glob("*.json"))
BROKEN_BASE = WORKFLOW_DIR / "Check a tag list.json"
WIDGET_TYPES = {"BOOLEAN": bool, "INT": int, "FLOAT": int | float, "STRING": str}
README_LINK = re.compile(r"\]\((workflows/[^)]+\.json)\)")


@pytest.fixture(scope="module")
def node_classes():
    # Replicates core's loader, since pytest cannot import the hyphenated pack root as a package.
    module_name = "ComfyUI-LogitTagger_workflows_test"
    spec = importlib.util.spec_from_file_location(module_name, PACK_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module.NODE_CLASS_MAPPINGS
    finally:
        del sys.modules[module_name]


def load_workflow(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pack_nodes(workflow, node_classes):
    return [node for node in workflow["nodes"] if node["type"] in node_classes]


def slot(entries, index):
    """The entry at a link's slot index, or an empty entry when the slot does not exist."""
    return entries[index] if 0 <= index < len(entries) else {}


def is_widget(spec):
    input_type = spec[0]
    options = spec[1] if len(spec) > 1 else {}

    if isinstance(input_type, list | tuple):
        return True
    return input_type in WIDGET_TYPES and not options.get("forceInput", False)


def widget_inputs(node_class):
    """The class's widget inputs as (name, type) pairs, in the order core lays out widgets_values."""
    input_types = node_class.INPUT_TYPES()
    sections = (input_types.get("required", {}), input_types.get("optional", {}))

    return [(name, spec[0]) for section in sections for name, spec in section.items() if is_widget(spec)]


def value_matches(input_type, value):
    if isinstance(input_type, list | tuple):
        return value in input_type
    return isinstance(value, WIDGET_TYPES[input_type])


def structure_problems(workflow, node_classes):
    node_types = {node["type"] for node in workflow["nodes"]}
    problems = []

    if not node_types & set(node_classes):
        problems.append(f"no node of a pack type among {sorted(node_types)}")
    if "Note" not in node_types:
        problems.append("no Note node")
    return problems


def link_end_problems(link, nodes):
    link_id, from_node, from_slot, to_node, to_slot, _ = link
    problems = []

    if from_node not in nodes or to_node not in nodes:
        return [f"link {link_id} names node {from_node} or node {to_node}, and one of them does not exist"]
    if slot(nodes[to_node].get("inputs", []), to_slot).get("link") != link_id:
        problems.append(f"link {link_id}: input slot {to_slot} of node {to_node} does not carry it")
    if link_id not in (slot(nodes[from_node].get("outputs", []), from_slot).get("links") or []):
        problems.append(f"link {link_id}: output slot {from_slot} of node {from_node} does not list it")
    return problems


def link_problems(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    link_ids = {link[0] for link in workflow["links"]}
    linked_inputs = [(node["id"], entry) for node in nodes.values() for entry in node.get("inputs", [])
                     if entry.get("link") is not None]
    problems = []

    for link in workflow["links"]:
        problems.extend(link_end_problems(link, nodes))
    problems.extend(f"node {node_id} input {entry['name']} names link {entry['link']}, which does not exist"
                    for node_id, entry in linked_inputs if entry["link"] not in link_ids)
    return problems


def widget_problems(workflow, node_classes):
    problems = []

    for node in pack_nodes(workflow, node_classes):
        label = f"node {node['id']} {node['type']}"
        widgets = widget_inputs(node_classes[node["type"]])
        values = node.get("widgets_values") or []
        if len(values) != len(widgets):
            problems.append(f"{label}: {len(values)} widget values for the widgets {[name for name, _ in widgets]}")
            continue
        problems.extend(f"{label}: widget {name} holds {value!r}, which is not a {input_type} value"
                        for (name, input_type), value in zip(widgets, values, strict=True)
                        if not value_matches(input_type, value))
    return problems


def wiring_problems(workflow, node_classes):
    problems = []

    for node in pack_nodes(workflow, node_classes):
        required = node_classes[node["type"]].INPUT_TYPES()["required"]
        must_link = ["clip", "image"] if "image" in required else ["clip"]
        linked = {entry["name"] for entry in node.get("inputs", []) if entry.get("link") is not None}
        problems.extend(f"node {node['id']} {node['type']}: {name} is not linked"
                        for name in must_link if name not in linked)
    return problems


def readme_workflow_names(readme_text):
    """The file names the README's table rows link under workflows/, with %20 read as a space."""
    table_rows = [line for line in readme_text.splitlines() if line.startswith("|")]

    return {unquote(target.rsplit("/", 1)[-1]) for row in table_rows for target in README_LINK.findall(row)}


def broken_copy(tmp_path, break_workflow):
    """A copy of a working workflow, broken by break_workflow, written to tmp_path and read back the way the checks
    read the shipped files."""
    workflow = load_workflow(BROKEN_BASE)
    path = tmp_path / BROKEN_BASE.name

    break_workflow(workflow)
    path.write_text(json.dumps(workflow), encoding="utf-8")
    return load_workflow(path)


def tag_check_node(workflow):
    return next(node for node in workflow["nodes"] if node["type"] == "LogitTagCheck")


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=lambda path: path.stem)
def test_workflow_holds_a_pack_node_and_a_note(path, node_classes):
    assert structure_problems(load_workflow(path), node_classes) == []


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=lambda path: path.stem)
def test_workflow_links_match_their_node_slots(path):
    assert link_problems(load_workflow(path)) == []


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=lambda path: path.stem)
def test_workflow_widget_values_match_the_pack_classes(path, node_classes):
    assert widget_problems(load_workflow(path), node_classes) == []


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=lambda path: path.stem)
def test_workflow_links_the_clip_and_a_required_image(path, node_classes):
    assert wiring_problems(load_workflow(path), node_classes) == []


def test_readme_table_links_every_workflow_file():
    readme_text = (PACK_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme_workflow_names(readme_text) == {path.name for path in WORKFLOW_PATHS}


def test_readme_comparison_passes_on_a_table_of_every_file(tmp_path):
    rows = [f"| [{path.stem}](workflows/{quote(path.name)}) | Runs {path.stem}. |" for path in WORKFLOW_PATHS]
    readme = tmp_path / "README.md"

    readme.write_text("\n".join(["| Workflow | What it does |", "|---|---|", *rows]), encoding="utf-8")

    assert readme_workflow_names(readme.read_text(encoding="utf-8")) == {path.name for path in WORKFLOW_PATHS}


def test_readme_comparison_fails_on_a_missing_row():
    rows = [f"| [{path.stem}](workflows/{quote(path.name)}) | Runs {path.stem}. |" for path in WORKFLOW_PATHS[1:]]

    assert readme_workflow_names("\n".join(rows)) != {path.name for path in WORKFLOW_PATHS}


def test_structure_check_fails_without_a_note(tmp_path, node_classes):
    def drop_note(workflow):
        workflow["nodes"] = [node for node in workflow["nodes"] if node["type"] != "Note"]

    assert structure_problems(broken_copy(tmp_path, drop_note), node_classes) == ["no Note node"]


def test_link_check_fails_on_a_changed_link_id(tmp_path):
    def change_link_id(workflow):
        workflow["links"][0][0] = 999

    problems = link_problems(broken_copy(tmp_path, change_link_id))

    assert "link 999: input slot 0 of node 4 does not carry it" in problems
    assert "link 999: output slot 0 of node 1 does not list it" in problems
    assert "node 4 input clip names link 1, which does not exist" in problems


def test_widget_check_fails_on_a_removed_widget_value(tmp_path, node_classes):
    def drop_widget_value(workflow):
        tag_check_node(workflow)["widgets_values"].pop()

    problems = widget_problems(broken_copy(tmp_path, drop_widget_value), node_classes)

    assert problems == ["node 4 LogitTagCheck: 1 widget values for the widgets ['strict', 'positions']"]


def test_widget_check_fails_on_a_widget_value_of_the_wrong_type(tmp_path, node_classes):
    def mistype_widget_value(workflow):
        tag_check_node(workflow)["widgets_values"][0] = "false"

    problems = widget_problems(broken_copy(tmp_path, mistype_widget_value), node_classes)

    assert problems == ["node 4 LogitTagCheck: widget strict holds 'false', which is not a BOOLEAN value"]


def test_wiring_check_fails_on_an_unlinked_clip(tmp_path, node_classes):
    def unlink_clip(workflow):
        next(entry for entry in tag_check_node(workflow)["inputs"] if entry["name"] == "clip")["link"] = None

    problems = wiring_problems(broken_copy(tmp_path, unlink_clip), node_classes)

    assert problems == ["node 4 LogitTagCheck: clip is not linked"]
