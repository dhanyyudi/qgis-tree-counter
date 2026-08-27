"""Indonesian translation completeness, loading, and fallback."""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "tree_counter"
TS_PATH = PACKAGE / "i18n" / "tree_counter_id.ts"

RUNTIME_DIALOG_BODY_TEMPLATES = {
    "State: {state}",
    "Python: {version}",
    "Platform: {platform}",
    "{name}: {versions}",
    "- {reason}",
    "{action} the Tree Counter runtime?",
    "{kind}: {title} - about {size} from {source}",
    "- {title} from {source} (about {size})",
    "Location: {location}",
    "The installed runtime will be deleted.",
    "The existing runtime is kept until the new one is verified.",
    "The installed runtime was built for a different platform.",
    "The host Python version is outside the supported range.",
    "The Python version changed since the runtime was installed.",
    "Required runtime files are missing.",
    "The runtime contains unknown components: {components}.",
    "The runtime could not import {module}.",
    "{component} no longer provides: {accelerators}.",
    "A runtime update is available for: {components}.",
}


def _ts_translations() -> dict[str, str]:
    tree = ET.parse(str(TS_PATH))
    result: dict[str, str] = {}
    for message in tree.iter("message"):
        source = message.find("source")
        translation = message.find("translation")
        if source is None or translation is None:
            continue
        value = translation.text or ""
        if translation.get("type") == "unfinished":
            value = ""
        result[source.text or ""] = value
    return result


def _tr_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tr"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append(node.args[0].value)
    return found


def _message_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "message"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value
            ):
                found.append(keyword.value.value)
    return found


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    return [node.target]


def _safe_message_values(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        for target in _assignment_targets(node):
            if isinstance(target, ast.Name) and target.id == "SAFE_MESSAGES":
                if isinstance(node.value, ast.Dict):
                    return [
                        item.value
                        for item in node.value.values
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    ]
    return []


def _constant_strings(path: Path, names: set[str]) -> set[str]:
    """Return the displayed strings held by the named module constants."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        matched = any(
            isinstance(target, ast.Name) and target.id in names
            for target in _assignment_targets(node)
        )
        if not matched:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.add(value.value)
        elif isinstance(value, ast.Tuple):
            for element in value.elts:
                if (
                    isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                ):
                    found.add(element.value)
                elif (
                    isinstance(element, ast.Tuple)
                    and element.elts
                    and isinstance(element.elts[0], ast.Constant)
                    and isinstance(element.elts[0].value, str)
                ):
                    found.add(element.elts[0].value)
        elif isinstance(value, ast.Dict):
            for item in value.values:
                if isinstance(item, ast.Constant) and isinstance(
                    item.value, str
                ):
                    found.add(item.value)
    return found


def _visible_sources() -> set[str]:
    sources: set[str] = set()
    for name in ("dock.py", "widgets.py", "runtime_dialog.py"):
        sources.update(_tr_literals(PACKAGE / "ui" / name))
    sources.update(_message_literals(PACKAGE / "ui" / "controller.py"))
    sources.update(_safe_message_values(PACKAGE / "errors.py"))
    sources.update(
        _constant_strings(
            PACKAGE / "ui" / "dock.py",
            {"DOCK_TITLE", "SECTION_TITLES", "SCOPE_CHOICES"},
        )
    )
    sources.update(
        _constant_strings(
            PACKAGE / "ui" / "runtime_dialog.py",
            {"DIALOG_TITLE", "ACTION_LABELS"},
        )
    )
    return sources


def _missing_translations(
    translations: dict[str, str], sources: set[str]
) -> list[str]:
    return sorted(source for source in sources if not translations.get(source))


def test_every_visible_string_has_an_indonesian_translation() -> None:
    translations = _ts_translations()
    missing = _missing_translations(translations, _visible_sources())

    assert missing == [], (
        "missing Indonesian translations: "
        + ", ".join(repr(item) for item in missing)
    )


def test_runtime_body_templates_are_collected() -> None:
    collected = set(_tr_literals(PACKAGE / "ui" / "runtime_dialog.py"))

    assert RUNTIME_DIALOG_BODY_TEMPLATES <= collected


def test_a_missing_runtime_body_translation_is_reported() -> None:
    translations = _ts_translations()
    translations.pop("State: {state}", None)

    assert _missing_translations(
        translations, RUNTIME_DIALOG_BODY_TEMPLATES
    ) == ["State: {state}"]


def test_no_translation_entry_is_unused() -> None:
    translations = _ts_translations()
    sources = _visible_sources()
    unused = sorted(source for source in translations if source not in sources)

    assert unused == [], (
        "translation entries with no visible source: "
        + ", ".join(repr(item) for item in unused)
    )


def test_english_remains_the_source_language() -> None:
    assert TS_PATH.is_file()
    assert not (PACKAGE / "i18n" / "tree_counter_en.ts").exists()


def test_the_ts_has_no_empty_translations() -> None:
    translations = _ts_translations()

    assert translations, "the translation file has no entries"
    empty = sorted(key for key, value in translations.items() if not value)
    assert empty == []


def test_indonesian_locale_translates_a_known_string(qgis_application) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator, tr

    app = QCoreApplication.instance()
    translator = install_translator(app, locale="id_ID")
    assert translator is not None
    try:
        assert tr("Start counting") == "Mulai menghitung"
    finally:
        app.removeTranslator(translator)


def test_other_locales_stay_english(qgis_application) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter.i18n import install_translator, tr

    app = QCoreApplication.instance()

    assert install_translator(app, locale="en_US") is None
    assert tr("Start counting") == "Start counting"


def test_a_missing_qm_falls_back_to_english(
    qgis_application, monkeypatch
) -> None:
    from qgis.PyQt.QtCore import QCoreApplication

    from tree_counter import i18n
    from tree_counter.i18n import install_translator, tr

    app = QCoreApplication.instance()
    monkeypatch.setattr(i18n, "qm_path", lambda: Path("/nonexistent/x.qm"))

    assert install_translator(app, locale="id_ID") is None
    assert tr("Start counting") == "Start counting"
