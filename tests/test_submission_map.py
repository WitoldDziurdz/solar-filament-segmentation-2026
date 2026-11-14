"""The submission map at the top of README.md and of the notebook, the licence, and the data statement.

The map is the table under the heading "## Kaggle submissions", the first section of
``README.md``; the same table opens the notebook's first markdown cell. It has one row per
Kaggle submission this repository rebuilds: the submission id, its public score, the
configuration that made it, a one-line recipe, its local report-half PQ with TP / FP / FN (or
"LB-only" when no validation image is held out from it), and the stage-2 setting of the
notebook that rebuilds the CSV. What these tests hold the map to:

* every published configuration (:data:`filament.pipeline.CONFIGURATIONS`) has exactly one
  row, and each submission id appears once;
* each row's weight files are the ones ``weights/manifest.json`` lists for that
  configuration, detector first, and its detector, input size, NMS IoU, training images and
  refiner alpha are the configuration's;
* each local score is the one notebook stage 2 records for that configuration
  (``VALIDATION_RECORDED``), and the "LB-only" rows are exactly the ones stage 7 does not
  score;
* the map agrees with the README's own table of measurements (*Published configurations*):
  the same PQ, TP / FP / FN, public score and submission id;
* the rebuild setting is a line of the notebook's stage 2, which ships with the default
  configuration;
* the final selections are left open, in both places, until they are made.

They also check that ``LICENSE`` is the GNU Affero General Public License v3 text, unchanged
from the copy Ultralytics distributes, and that the README states the repository's licence
and the data the models were trained on.
"""

import ast
import hashlib
import importlib.metadata
import json
import re

import pytest

from filament import pipeline

from .conftest import REPO

README = REPO / "README.md"
NOTEBOOK = REPO / "notebooks" / "pipeline.ipynb"
LICENSE = REPO / "LICENSE"
MANIFEST = REPO / "weights" / "manifest.json"

TITLE = "# Solar filament instance segmentation"
MAP_HEADING = "## Kaggle submissions"
MAP_HEADER = ["Kaggle submission", "Public score", "Configuration", "Recipe",
              "Local PQ, report half (TP / FP / FN)", "Rebuilt by (notebook stage 2)"]
FINAL_SELECTIONS = "Final selections: to be filled in when the user selects them (2026-11-14)."
LB_ONLY = "LB-only: detector trained on all 707 training images"

# LICENSE is Ultralytics 8.4.126's licences/LICENSE, byte for byte (the package is AGPL-3.0).
LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
# The same text with every run of whitespace collapsed to one space. The copy that
# ultralytics-thop 2.2.5 distributes has this digest too; the two files differ only in where
# two lines of the appendix's sample notice ("How to Apply These Terms") are wrapped.
LICENSE_WORDS_SHA256 = "2cd0fb7883a3b3553dedbb0bad171646d46f51e2642951aae7c59cb5cec86c46"
AGPL_SECTIONS = [
    "Definitions.", "Source Code.", "Basic Permissions.",
    "Protecting Users' Legal Rights From Anti-Circumvention Law.", "Conveying Verbatim Copies.",
    "Conveying Modified Source Versions.", "Conveying Non-Source Forms.", "Additional Terms.",
    "Termination.", "Acceptance Not Required for Having Copies.",
    "Automatic Licensing of Downstream Recipients.", "Patents.", "No Surrender of Others' Freedom.",
    "Remote Network Interaction; Use with the GNU General Public License.",
    "Revised Versions of this License.", "Disclaimer of Warranty.", "Limitation of Liability.",
    "Interpretation of Sections 15 and 16.",
]


# --------------------------------------------------------------------------
# Reading the map
# --------------------------------------------------------------------------

def readme():
    return README.read_text(encoding="utf-8")


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def first_cell():
    cell = notebook()["cells"][0]
    assert cell["cell_type"] == "markdown"
    return "".join(cell["source"])


def table_after(markdown, heading):
    """The lines of the first markdown table after the line ``heading``."""
    lines = markdown.splitlines()
    assert lines.count(heading) == 1, f"{heading!r} must appear exactly once"
    table = []
    for line in lines[lines.index(heading) + 1:]:
        if line.startswith("|"):
            table.append(line)
        elif table or line.startswith("#"):
            break
    assert table, f"no table under {heading!r}"
    return table


def cells(line):
    assert line.startswith("|") and line.endswith("|"), line
    return [cell.strip() for cell in line[1:-1].split("|")]


def parse_map(markdown):
    """``[{column: text}, ...]``, one per row of the map; the header is checked first."""
    header, rule, *rows = table_after(markdown, MAP_HEADING)
    assert cells(header) == MAP_HEADER
    assert set("".join(cells(rule))) <= set("-:")
    return [dict(zip(MAP_HEADER, cells(row), strict=True)) for row in rows]


def name_of(row):
    """The configuration name of a row: ``"013ft"`` from ``"013ft"`` or ``"013"` (default)``."""
    found = re.fullmatch(r'`"([0-9a-z]+)"`(?: \(default\))?', row["Configuration"])
    assert found, row["Configuration"]
    return found.group(1)


def local_score(row):
    """``{"pq", "tp", "fp", "fn"}`` of a row, or None for an "LB-only" row."""
    text = row[MAP_HEADER[4]]
    if text == LB_ONLY:
        return None
    found = re.fullmatch(r"(\d\.\d{4}) \((\d+) / (\d+) / (\d+)\)", text)
    assert found, text
    pq, tp, fp, fn = found.groups()
    return {"pq": float(pq), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def manifest_files(name):
    """``[detector file, refiner file]`` that the manifest lists for configuration ``name``."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = (manifest["files"] if name == pipeline.DEFAULT_CONFIGURATION
               else manifest["configurations"][name]["files"])
    by_role = {entry["role"]: entry for entry in entries}
    return [by_role["detector"], by_role["refiner"]]


def stage_2():
    """The code cell of notebook stage 2: the one that assigns ``CONFIGURATION``."""
    found = [cell for cell in notebook()["cells"] if cell["cell_type"] == "code"
             and re.search(r"^CONFIGURATION = ", "".join(cell["source"]), re.M)]
    assert len(found) == 1
    return "".join(found[0]["source"])


def assigned(source, variable):
    """The literal value assigned to ``variable`` at the top level of ``source``."""
    tree = ast.parse(source)
    (value,) = [node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)]
    return ast.literal_eval(value)


@pytest.fixture(scope="module")
def rows():
    return parse_map(readme())


# --------------------------------------------------------------------------
# Where the map is
# --------------------------------------------------------------------------

def paragraphs_before_the_map(markdown):
    """The paragraphs between the title line and the map's heading."""
    lines = markdown.splitlines()
    between = "\n".join(lines[1:lines.index(MAP_HEADING)])
    return [block for block in between.split("\n\n") if block.strip()]


def test_the_readme_opens_with_the_map():
    lines = readme().splitlines()
    assert lines[0] == TITLE
    sections = [line for line in lines if line.startswith("## ")]
    assert sections[0] == MAP_HEADING
    assert len(paragraphs_before_the_map(readme())) <= 1            # the title's one-paragraph introduction


def test_the_notebook_opens_with_the_same_map():
    lines = first_cell().splitlines()
    assert lines[0].startswith(TITLE)
    assert [line for line in lines if line.startswith("## ")][0] == MAP_HEADING
    assert paragraphs_before_the_map(first_cell()) == []
    assert table_after(first_cell(), MAP_HEADING) == table_after(readme(), MAP_HEADING)


def test_the_final_selections_are_open_or_rows_of_the_map(rows):
    """One line in each place, the same in both: open until the selections are made, then two rows of the map.

    Filled in, it reads ``**Final selections: 58242720 (`"013ft"`) and 58242724 (`"035"`).**`` (example only).
    """
    marked = [[line for line in text.splitlines() if "Final selections" in line] for text in (readme(), first_cell())]
    assert len(marked[0]) == 1 and marked[0] == marked[1]
    if marked[0][0] == f"**{FINAL_SELECTIONS}**":
        return
    filled = re.fullmatch(r'\*\*Final selections: (\d{8}) \(`"([0-9a-z]+)"`\) and (\d{8}) \(`"([0-9a-z]+)"`\)\.\*\*',
                          marked[0][0])
    assert filled, f"neither open nor two map rows: {marked[0][0]!r}"
    in_map = {(row["Kaggle submission"], name_of(row)) for row in rows}
    chosen = {(filled.group(1), filled.group(2)), (filled.group(3), filled.group(4))}
    assert len(chosen) == 2 and chosen <= in_map


# --------------------------------------------------------------------------
# The map against the package
# --------------------------------------------------------------------------

def test_every_published_configuration_has_one_row(rows):
    names = [name_of(row) for row in rows]
    assert sorted(names) == sorted(pipeline.CONFIGURATIONS)
    submissions = [row["Kaggle submission"] for row in rows]
    assert all(re.fullmatch(r"\d{8}", submission) for submission in submissions)
    assert len(set(submissions)) == len(submissions)
    assert all(re.fullmatch(r"0\.\d\d", row["Public score"]) for row in rows)
    defaults = [row["Configuration"] for row in rows if row["Configuration"].endswith("(default)")]
    assert defaults == [f'`"{pipeline.DEFAULT_CONFIGURATION}"` (default)']


def test_the_weight_files_are_the_manifests(rows):
    for row in rows:
        listed = [entry["name"] for entry in manifest_files(name_of(row))]
        assert re.findall(r"`([^`]+\.pt)`", row["Recipe"]) == listed, name_of(row)
        assert all((REPO / "weights" / file).is_file() for file in listed)


def test_the_recipe_is_the_configurations(rows):
    default = pipeline.CONFIGURATIONS[pipeline.DEFAULT_CONFIGURATION]
    for row in rows:
        name, recipe = name_of(row), row["Recipe"]
        configuration = pipeline.CONFIGURATIONS[name]
        detector_entry, refiner_entry = manifest_files(name)
        alpha = re.findall(r"alpha (\d+(?:\.\d+)?)", recipe)
        assert [float(value) for value in alpha] == [configuration.refiner.alpha], name
        assert re.findall(r"(\d{4}) px", recipe) == [str(configuration.detector.imgsz)], name
        architecture = re.match(r"YOLO11[lm]-seg", recipe).group(0)
        assert detector_entry["description"].startswith(architecture + ","), name
        nms = re.findall(r"NMS IoU (\d\.\d)", recipe)
        if configuration.detector.nms_iou == default.detector.nms_iou:
            assert nms == [], name                         # only a non-default NMS IoU is named
        else:
            assert [float(value) for value in nms] == [configuration.detector.nms_iou], name
        all_images = "all 707 images" in recipe
        assert all_images == ("all 707 training images" in detector_entry["description"]), name
        assert all_images != ("folds 1-4" in recipe), name
        fine_tuned = "fine-tuned refiner" in recipe
        assert fine_tuned == refiner_entry["name"].endswith("_ft.pt"), name


def test_the_local_scores_are_the_notebooks_records(rows):
    recorded = assigned(stage_2(), "VALIDATION_RECORDED")
    for row in rows:
        name, score = name_of(row), local_score(row)
        if recorded[name] is None:
            assert score is None, f"{name} has no held-out images, so its row must say LB-only"
            assert "all 707 images" in row["Recipe"]
        else:
            assert score is not None, name
            assert (score["tp"], score["fp"], score["fn"]) == (
                recorded[name]["tp"], recorded[name]["fp"], recorded[name]["fn"]), name
            assert score["pq"] == round(recorded[name]["pq"], 4), name


def measurement_table():
    """The table of *Published configurations* with each configuration's PQ ... public score."""
    section = readme().split("\n## Published configurations\n", 1)[1].split("\n## ", 1)[0]
    header = ["", "PQ", "SQ", "RQ", "TP", "FP", "FN", "Public leaderboard"]
    lines = section.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("|") and cells(line) == header]
    assert len(starts) == 1
    table = []
    for line in lines[starts[0] + 2:]:
        if not line.startswith("|"):
            break
        table.append(line)
    return table


def test_the_map_agrees_with_the_measurement_table(rows):
    measured = {}
    for line in measurement_table():
        first, *rest = cells(line)
        if re.fullmatch(r'`"[0-9a-z]+"`', first):
            measured[first.strip('`"')] = rest
    assert sorted(measured) == sorted(pipeline.CONFIGURATIONS)
    for row in rows:
        name, score = name_of(row), local_score(row)
        pq, _sq, _rq, tp, fp, fn, public = measured[name]
        assert public == f"{row['Public score']} (submission {row['Kaggle submission']})", name
        if score is None:
            assert pq.startswith("no local score"), name
        else:
            assert (float(pq.strip("*")), int(tp), int(fp), int(fn)) == (
                score["pq"], score["tp"], score["fp"], score["fn"]), name


def test_the_rebuild_setting_is_a_line_of_stage_2(rows):
    source = stage_2()
    assert assigned(source, "CONFIGURATION") == pipeline.DEFAULT_CONFIGURATION    # shipped as the default
    assert len(re.findall(r"^CONFIGURATION = ", source, re.M)) == 1
    for row in rows:
        name = name_of(row)
        shipped = " (as shipped)" if name == pipeline.DEFAULT_CONFIGURATION else ""
        assert row["Rebuilt by (notebook stage 2)"] == f'`CONFIGURATION = "{name}"`{shipped}'


# --------------------------------------------------------------------------
# Licence and data
# --------------------------------------------------------------------------

def test_the_licence_is_the_gnu_agpl_v3():
    raw = LICENSE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LICENSE_SHA256
    text = raw.decode("ascii")
    assert hashlib.sha256(" ".join(text.split()).encode("ascii")).hexdigest() == LICENSE_WORDS_SHA256
    lines = text.splitlines()
    assert lines[0].strip() == "GNU AFFERO GENERAL PUBLIC LICENSE"
    assert lines[1].strip() == "Version 3, 19 November 2007"
    sections = re.findall(r"^  (\d{1,2})\. (.+)$", text, re.M)
    assert sections == [(str(number), title) for number, title in enumerate(AGPL_SECTIONS)]
    assert text.count("END OF TERMS AND CONDITIONS") == 1
    assert "How to Apply These Terms to Your New Programs" in text


def installed_licence(distribution):
    """The licence file a distribution installed, or None when it is not installed."""
    try:
        files = importlib.metadata.distribution(distribution).files or []
    except importlib.metadata.PackageNotFoundError:
        return None
    found = [path for path in files if path.name == "LICENSE" and ".dist-info" in str(path)]
    return found[0].locate().read_bytes() if found else None


def test_the_licence_matches_the_copies_ultralytics_installs():
    ultralytics, thop = installed_licence("ultralytics"), installed_licence("ultralytics-thop")
    if ultralytics is None or thop is None:
        pytest.skip("ultralytics or ultralytics-thop is not installed")
    assert LICENSE.read_bytes() == ultralytics
    assert " ".join(thop.decode("ascii").split()) == " ".join(LICENSE.read_text("ascii").split())


def test_the_readme_states_the_licence():
    section = readme().split("\n## Licence\n", 1)[1].split("\n## ", 1)[0]
    for phrase in ("GNU Affero General Public License v3.0", "AGPL-3.0", "`LICENSE`", "Ultralytics",
                   "not redistributed"):
        assert phrase in section, phrase
    assert "Not chosen yet" not in readme()


def test_the_readme_states_the_training_data():
    text = " ".join(readme().split())
    for phrase in ("MAGFiLO_1.0_Annotations_kaggle2026_train.json", "No other data was used",
                   "no other release or part of MAGFiLO", "no other annotations", "no test labels",
                   "no pseudo-labels on the test images"):
        assert phrase in text, phrase
