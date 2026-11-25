"""Automated tests for the de-identification pipeline.

Run:  python -m pytest -q
      python -m pytest -q -m "not gliner"     # skip the (slower) GLiNER tests

Uses the made-up roster in examples/test_roster.csv and a temporary student map,
so your private roster.csv and student_map.csv are never read or changed.
Student numbers follow the test roster order:
  1 Jordan Nakamura (jnakamu3, 48213377)   2 Priya Raghunathan (praghuna, 51902264)
  3 Alex Long (along4, 40055501)           4 José García (jgarcia9, 40055502)
"""
import csv
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import deid  # noqa: E402
import llm_check  # noqa: E402

TEST_ROSTER = ROOT / "examples" / "test_roster.csv"
P = r"<[A-Z_]+_\d+>"  # any numbered non-roster placeholder, e.g. <PERSON_3>


@pytest.fixture(autouse=True)
def test_roster(tmp_path, monkeypatch):
    """Every test starts with the example roster, spaCy, and a throwaway student map."""
    monkeypatch.setattr(deid, "MAP_PATH", tmp_path / "student_map.csv")
    monkeypatch.setattr(deid, "OUTPUT_DIR", tmp_path / "output")
    deid.set_ner("spacy")
    deid.configure(roster_path=TEST_ROSTER, persist=False)
    yield


def check(text, expected):
    """expected: exact string, or a compiled regex for outputs with numbered placeholders."""
    out = deid.deidentify(text)
    if isinstance(expected, re.Pattern):
        assert expected.fullmatch(out), f"\n  in:  {text}\n  out: {out}"
    else:
        assert out == expected, f"\n  in:  {text}\n  out: {out}"


# ---------------------------------------------------------------------------
# 1. Roster: names, UCInetIDs, UCI IDs, emails
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("# author: jordan nakamura", "# author: <STUDENT_1>"),
    ("Jordan Nakamura helped Priya Raghunathan; Nakamura found the bug.",
     "<STUDENT_1> helped <STUDENT_2>; <STUDENT_1> found the bug."),
    ("NAKAMURA wrote this", "<STUDENT_1> wrote this"),
    ("# hw3 - praghuna - 51902264", "# hw3 - <STUDENT_2_NETID> - <STUDENT_2_UCIID>"),
    ("contact jnakamu3@uci.edu", "contact <STUDENT_1_EMAIL>"),
    ("Raghunathan's code", "<STUDENT_2>'s code"),
])
def test_roster_matching(text, expected):
    check(text, expected)


# ---------------------------------------------------------------------------
# 2. Roster names / UCInetIDs inside code identifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("result = nakamuraSolution(data)", "result = <STUDENT_1>Solution(data)"),
    ("def solveNakamura(x): pass", "def solve<STUDENT_1>(x): pass"),
    ("NAKAMURA_MAX = 10", "<STUDENT_1>_MAX = 10"),
    ("open('jnakamu3_hw3.py')", "open('<STUDENT_1_NETID>_hw3.py')"),
    ("from raghunathan_utils import helper", "from <STUDENT_2>_utils import helper"),
    ("class MyNakamuraTree: pass", "class My<STUDENT_1>Tree: pass"),
])
def test_names_in_code(text, expected):
    check(text, expected)


@pytest.mark.parametrize("text", [
    "def sleep(): pass  # kitchenSink helper",   # "lee", "chen" hidden inside words
    "acetone = 3; leeway = 2",
])
def test_names_in_code_no_false_positives(text):
    check(text, text)


# ---------------------------------------------------------------------------
# 3. Surnames that are common words (Long) are only matched as full names
# ---------------------------------------------------------------------------
def test_common_word_surname_full_name_only():
    check("Alex Long wrote this. long total = 0L; Long.parseLong(s); a long list",
          "<STUDENT_3> wrote this. long total = 0L; Long.parseLong(s); a long list")


def test_common_word_surname_netid_still_matched():
    check("submitted by along4", "submitted by <STUDENT_3_NETID>")


# ---------------------------------------------------------------------------
# 4. Accents: "José García" on the roster matches however it is typed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("Thanks José García!", "Thanks <STUDENT_4>!"),
    ("Thanks Jose Garcia!", "Thanks <STUDENT_4>!"),
    ("garcia said hi", "<STUDENT_4> said hi"),
    ("see garciaSolver()", "see <STUDENT_4>Solver()"),
])
def test_accents(text, expected):
    check(text, expected)


# ---------------------------------------------------------------------------
# 5. Regex rules: emails, phones, IPs, paths, UCI IDs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("email me at someone.else@gmail.com", re.compile(r"email me at <EMAIL_ADDRESS_\d+>")),
    ("call 949-555-0192 tonight", re.compile(r"call <PHONE_NUMBER_\d+> tonight")),
    ("call (949) 555-0192 tonight", re.compile(r"call <PHONE_NUMBER_\d+> tonight")),
    ("ssh to 10.0.0.12", re.compile(r"ssh to <IP_ADDRESS_\d+>")),
    (r"C:\Users\mstone\Desktop\hw3\main.py", re.compile(r"C:\\Users\\<USERNAME_\d+>\\Desktop\\hw3\\main\.py")),
    ('open("C:\\\\Users\\\\mstone\\\\data.txt")', re.compile(r'open\("C:\\\\Users\\\\<USERNAME_\d+>\\\\data\.txt"\)')),
    ("/Users/mstone/Documents/lab2.py", re.compile(r"/Users/<USERNAME_\d+>/Documents/lab2\.py")),
    ("/home/mstone/proj/run.sh", re.compile(r"/home/<USERNAME_\d+>/proj/run\.sh")),
    (r"C:\Users\jnakamu3\hw3", r"C:\Users\<STUDENT_1_NETID>\hw3"),   # roster wins over the path rule
    ("my repo is github.com/carlos-mendez-dev/lab3",
     re.compile(r"my repo is github\.com/<USERNAME_\d+>/lab3")),
    ("https://www.linkedin.com/in/maya.r.chen.", re.compile(r"https://www\.linkedin\.com/in/<USERNAME_\d+>\.")),
    ("see github.com/jnakamu3", "see github.com/<STUDENT_1_NETID>"),
    ("my student ID is 77665544", re.compile(r"my student ID is <UCI_ID_\d+>")),
    ("UCI ID: 77665544", re.compile(r"UCI ID: <UCI_ID_\d+>")),
    ("id number 77665544", re.compile(r"id number <UCI_ID_\d+>")),
])
def test_regex_rules(text, expected):
    check(text, expected)


# ---------------------------------------------------------------------------
# 6. Names the model finds (not on the roster), and consistent labels
# ---------------------------------------------------------------------------
def test_unknown_names_get_consistent_labels():
    out = deid.deidentify("I asked Dr. Emily Watson, then emailed Emily Watson again.")
    labels = re.findall(P, out)
    assert "Emily" not in out and "Watson" not in out, out
    assert len(labels) == 2 and labels[0] == labels[1], out


@pytest.mark.parametrize("text, expected", [
    # Regression: spaCy once tagged "praghuna\ndef average(nums" as ONE name and erased code.
    ("# author: praghuna\ndef average(nums):\n    return sum(nums) / len(nums)",
     "# author: <STUDENT_2_NETID>\ndef average(nums):\n    return sum(nums) / len(nums)"),
    ("# by Marcus Bell\nimport os\nprint(os.getcwd())",
     re.compile(r"# by <PERSON_\d+>\nimport os\nprint\(os\.getcwd\(\)\)")),
    # Where a model name overlaps a roster match, the roster label wins.
    ("Jordan Nakamura's solution", "<STUDENT_1>'s solution"),
])
def test_model_names_never_damage_code(text, expected):
    check(text, expected)


def test_labels_stable_across_calls():
    a = deid.deidentify("Hi, I'm Marcus Bell.")
    b = deid.deidentify("Marcus Bell again here.")
    assert re.findall(P, a) == re.findall(P, b), (a, b)


# ---------------------------------------------------------------------------
# 7. Things that must NOT change: everyday code, errors and questions
# ---------------------------------------------------------------------------
CLEAN = [
    "def compute_average(scores):\n    return sum(scores) / len(scores)",
    "for user in users:\n    print(user.name, user.email)",
    "class Person:\n    def __init__(self, name):\n        self.name = name",
    "student = Student(name='', age=0)",
    "author = input('Enter author name: ')",
    "import pandas as pd\ndf = pd.read_csv('grades.csv')",
    "public static void main(String[] args) { long count = 0; }",
    "const total = items.reduce((a, b) => a + b, 0);",
    "SELECT name, grade FROM students WHERE grade > 90;",
    "Traceback (most recent call last):\n  File \"main.py\", line 12, in <module>\nIndexError: list index out of range",
    "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
    "How do I reverse a linked list in place?",
    "Why does my recursion hit the max depth on large inputs?",
    "Can you explain what a hash map is?",
    "timestamp = 20261001  # 8 digits but not an ID",
    "x = 42; y = 3.14159; port = 8080",
    "The Fibonacci sequence starts 0, 1, 1, 2, 3",
    "Is the Turing machine on the final?",
    "Grace period for submissions ends Friday",
    "thanks, that worked!",
]


@pytest.mark.parametrize("text", CLEAN)
def test_clean_text_unchanged(text):
    check(text, text)


# Known spaCy weaknesses (documented limitations, not regressions).
@pytest.mark.xfail(reason="spaCy tags capitalized common words / famous names as PERSON", strict=False)
@pytest.mark.parametrize("text", [
    "Will this work? Mark the task as done",
    "Use Dijkstra's algorithm for shortest paths",
])
def test_known_spacy_false_positives(text):
    check(text, text)


# ---------------------------------------------------------------------------
# 8. Two students sharing a name -> <STUDENT_AMBIGUOUS>, unique IDs still resolve
# ---------------------------------------------------------------------------
def test_ambiguous_names(tmp_path):
    roster = tmp_path / "roster.csv"
    roster.write_text(
        "first_name,last_name,ucinetid,uci_id\n"
        "Kevin,Cetin,kcetin1,40000001\n"
        "Kevin,Cetin,kcetin2,40000002\n", encoding="utf-8")
    deid.configure(roster_path=roster, persist=False)
    check("Hey Kevin Cetin, see kcetin2_hw.py",
          "Hey <STUDENT_AMBIGUOUS>, see <STUDENT_2_NETID>_hw.py")


# ---------------------------------------------------------------------------
# 9. No roster: still catches emails, paths, and UCInetIDs harvested from @uci.edu emails
# ---------------------------------------------------------------------------
def test_no_roster_mode():
    texts = ["email me at tkwan5@uci.edu", "tkwan5_lab2.py crashes", "Hi, I'm Jordan Nakamura"]
    deid.configure(use_roster=False, extra_netids=deid.harvest_netids(texts), persist=False)
    assert deid.deidentify(texts[0]) == "email me at <STUDENT_1_EMAIL>"
    assert deid.deidentify(texts[1]) == "<STUDENT_1_NETID>_lab2.py crashes"
    assert re.fullmatch(r"Hi, I'm <PERSON_\d+>", deid.deidentify(texts[2]))


def test_harvest_netids_skips_staff_style_addresses():
    assert deid.harvest_netids(["a jdoe2@uci.edu b first.last@uci.edu c X@gmail.com"]) == ["jdoe2"]


def test_missing_roster_is_not_an_error(tmp_path):
    info = deid.configure(roster_path=tmp_path / "nope.csv", persist=False)
    assert info["roster_students"] == 0
    check("contact someone@gmail.com", re.compile(r"contact <EMAIL_ADDRESS_\d+>"))


# ---------------------------------------------------------------------------
# 10. LLM safety checks (no Ollama needed: the model call is faked)
# ---------------------------------------------------------------------------
def fake_llm(monkeypatch, findings):
    monkeypatch.setattr(llm_check, "_ask", lambda chunk: findings)


def test_llm_findings_are_validated(monkeypatch):
    text = "I'm <STUDENT_2_NETID>, the only transfer student from Izmir."
    fake_llm(monkeypatch, [
        {"text": "Izmir", "type": "detail", "is_personal_info": True, "reason": "hometown"},
        {"text": "STUDENT_2_NETID", "type": "x", "is_personal_info": True, "reason": "inside placeholder"},
        {"text": "Ankara", "type": "x", "is_personal_info": True, "reason": "not in text (invented)"},
        {"text": "transfer", "type": "x", "is_personal_info": False, "reason": "model said not PII"},
        {"text": "izmir", "type": "x", "is_personal_info": True, "reason": "duplicate"},
        {"text": "I", "type": "x", "is_personal_info": True, "reason": "too short"},
    ])
    assert [f["text"] for f in llm_check.llm_find_pii(text)] == ["Izmir"]


def test_mask_findings_never_touches_placeholders_and_whole_words_only():
    text = "<STUDENT_2_NETID> met Lee while asleep"
    findings = [{"text": "STUDENT_2_NETID", "type": "", "reason": ""},
                {"text": "Lee", "type": "", "reason": ""}]
    out, applied = llm_check.mask_findings(text, findings, lambda v: "<LLM_PII_1>")
    assert out == "<STUDENT_2_NETID> met <LLM_PII_1> while asleep"
    assert [f["text"] for f in applied] == ["Lee"]


def test_chunks_cover_long_text_exactly():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(300))
    assert "".join(llm_check._chunks(text)) == text


# ---------------------------------------------------------------------------
# 11. CSV end to end
# ---------------------------------------------------------------------------
def test_csv_end_to_end(tmp_path):
    src = tmp_path / "msgs.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "user_note", "message_content"])
        w.writerow(["1", "Jordan Nakamura", "Hi, I'm Jordan Nakamura. My code crashes."])
        w.writerow(["2", "", "# hw3\n# author: praghuna\ndef average(nums):\n    return sum(nums) / len(nums)"])
        w.writerow(["3", "", "How do I sort a list of tuples?"])
    deid.deidentify_csv(src, columns=["message_content"], roster_path=TEST_ROSTER)

    with open(deid.OUTPUT_DIR / "msgs_deid.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3                                         # no rows dropped
    assert rows[0]["user_note"] == "Jordan Nakamura"              # unscanned column untouched
    assert rows[0]["message_content"] == "Hi, I'm <STUDENT_1>. My code crashes."
    assert rows[1]["message_content"].splitlines()[1] == "# author: <STUDENT_2_NETID>"
    assert rows[1]["message_content"].count("\n") == 3            # code layout kept
    assert rows[2]["message_content"] == "How do I sort a list of tuples?"


# ---------------------------------------------------------------------------
# 12. GLiNER (optional model; skipped if not installed)
# ---------------------------------------------------------------------------
@pytest.mark.gliner
def test_gliner_names_and_code_filter():
    pytest.importorskip("gliner")
    deid.set_ner("gliner")
    deid.configure(use_roster=False, persist=False)
    out = deid.deidentify("thanks marcus, and hey @priya")
    assert "marcus" not in out and "priya" not in out, out
    for text in ["for user in users:\n    print(user.name)", "class Person:\n    pass",
                 "import pandas as pd\ndf = pd.read_csv('a.csv')"]:
        check(text, text)
