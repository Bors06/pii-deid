import argparse
import builtins
import csv
import keyword
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from wordfreq import zipf_frequency

from llm_check import check_ollama, llm_find_pii, mask_findings
from presidio_analyzer import (AnalyzerEngine, EntityRecognizer, Pattern,
                               PatternRecognizer, RecognizerResult)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

HERE = Path(__file__).parent
ROSTER_PATH = HERE / "roster.csv"
MAP_PATH = HERE / "student_map.csv"     # PRIVATE: student key -> student number
OUTPUT_DIR = HERE / "output"

csv.field_size_limit(2**31 - 1)  # allow very long cells (whole code files)


# ---------------------------------------------------------------------------
# 0. Helpers: accents and common words
# ---------------------------------------------------------------------------
def strip_accents(s):
    """"José García" -> "Jose Garcia" (so roster names match however they're typed)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Surnames / UCInetIDs that are also everyday words or code keywords (Long, King, Park,
# Price, Young...) are only matched as part of a full name. Otherwise a student named Long
# would mask every Java `long` and every "a long list".
COMMON_WORD_ZIPF = 5.0  # wordfreq scale: ~5 = very common word ("long" 5.8, "smith" 4.9)

CODE_WORDS = (
    {w.lower() for w in keyword.kwlist}
    | {w.lower() for w in dir(builtins) if w.isalpha()}
    | set("""
        abstract assert boolean break byte case catch char class const continue default do
        double else enum extends final finally float for goto if implements import instanceof
        int interface long native new package private protected public return short static
        super switch synchronized this throw throws transient try void volatile while var let
        function typeof undefined null true false await async yield delete export struct union
        unsigned signed sizeof typedef extern register auto inline string vector array list map
        set queue stack node tree graph key value item data main args test temp count total
        result answer select from where join group order having insert update grant revoke
        table index view create drop alter
    """.split())
)


def is_common_word(word):
    w = strip_accents(word).lower()
    return w in CODE_WORDS or zipf_frequency(w, "en") >= COMMON_WORD_ZIPF


# ---------------------------------------------------------------------------
# 1. Roster (optional) + stable student numbers
# ---------------------------------------------------------------------------
def read_roster_rows(path=ROSTER_PATH):
    """Return roster rows, or [] (with a warning) if there is no usable roster."""
    if not Path(path).exists():
        print(f"Note: no roster at {path}. Running without roster matching.")
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("uci_id") or "").strip()]
    if not rows:
        print(f"Note: roster at {path} has no usable rows. Running without roster matching.")
    return rows


def load_student_numbers(keys, persist=True):
    """Reuse saved numbers; give new students the next free number.

    A key is a UCI ID for roster students, or "netid:<id>" for UCInetIDs
    found in the data itself (students who aren't on a roster).
    """
    mapping = {}
    if MAP_PATH.exists():
        with open(MAP_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = r.get("key") or r.get("uci_id")   # older files used "uci_id"
                mapping[key] = int(r["student_num"])
    next_num = max(mapping.values(), default=0) + 1
    added = 0
    for key in keys:
        if key not in mapping:
            mapping[key] = next_num
            next_num += 1
            added += 1
    if added and persist:
        with open(MAP_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "student_num"])
            for key, num in sorted(mapping.items(), key=lambda x: x[1]):
                w.writerow([key, num])
        print(f"student_map.csv: {added} new student(s) assigned numbers")
    return mapping


def build_index(roster_rows, extra_netids=(), persist=True):
    """Detection lists + lookup (lowercase value -> {(student_num, kind)})."""
    roster_netids = {r["ucinetid"].strip().lower() for r in roster_rows if r["ucinetid"].strip()}
    extra = sorted({n.lower() for n in extra_netids} - roster_netids)

    keys = [r["uci_id"].strip() for r in roster_rows] + [f"netid:{n}" for n in extra]
    numbers = load_student_numbers(keys, persist=persist)

    names, netids, uci_ids = set(), set(), set()
    lookup = defaultdict(set)
    common_word_ids = 0  # surnames / UCInetIDs limited to full-name matching

    def add(value, num, kind, bucket=None):
        value = value.strip()
        if value:
            lookup[strip_accents(value).lower()].add((num, kind))
            if bucket is not None:
                bucket.add(value)
                bucket.add(strip_accents(value))  # "José García" also matches "Jose Garcia"

    def add_netid(netid, num):
        nonlocal common_word_ids
        if not netid:
            return
        add(f"{netid}@uci.edu", num, "EMAIL")          # the email is always safe to match
        if is_common_word(netid):
            common_word_ids += 1
        else:
            add(netid, num, "NETID", netids)

    for row in roster_rows:
        num = numbers[row["uci_id"].strip()]
        first = row["first_name"].strip()
        last = row["last_name"].strip()
        if first and last:
            add(f"{first} {last}", num, "NAME", names)
            add(f"{first.split()[0]} {last}", num, "NAME", names)
        if len(last) >= 3:
            if is_common_word(last):
                common_word_ids += 1                    # "Long": full name only
            else:
                add(last, num, "NAME", names)
        add_netid(row["ucinetid"].strip(), num)
        add(row["uci_id"], num, "UCIID", uci_ids)
        add(row.get("email") or "", num, "EMAIL")

    for netid in extra:  # UCInetIDs harvested from @uci.edu emails in the data
        add_netid(netid, numbers[f"netid:{netid}"])

    return sorted(names), sorted(netids), sorted(uci_ids), lookup, common_word_ids


# Fallback 2: UCInetIDs from @uci.edu addresses in the data itself.
# The lookbehind skips "first.last@uci.edu" (staff-style addresses aren't UCInetIDs).
UCI_EMAIL = re.compile(r"(?i)(?<![\w.%+-])([a-z][a-z0-9]{2,7})@uci\.edu\b")


def harvest_netids(texts):
    return sorted({m.group(1).lower() for t in texts if t for m in UCI_EMAIL.finditer(t)})


# ---------------------------------------------------------------------------
# 2. Recognizers (detection)
# ---------------------------------------------------------------------------
HOME_DIR_USERNAME = r"(?i)(?:(?<=users\\)|(?<=users\\\\)|(?<=users/)|(?<=home/))[^\\/\s\"'<>]+"

# The username part of profile links: github.com/<user>, linkedin.com/in/<user>, ...
PROFILE_URL_USERNAME = (
    r"(?i)(?:(?<=github\.com/)|(?<=gitlab\.com/)|(?<=linkedin\.com/in/)|(?<=twitter\.com/)"
    r"|(?<=x\.com/)|(?<=instagram\.com/)|(?<=leetcode\.com/u/)|(?<=replit\.com/@))"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*[A-Za-z0-9_]"
)


class UciIdContextRecognizer(EntityRecognizer):
    """Fallback 1: an 8-digit number counts as a UCI ID only right after words like
    "student ID", "UCI ID", "ID:", "id number" - so ordinary numbers in code are left alone."""

    PATTERN = re.compile(
        r"(?i)\b(?:(?:student|uci|campus)\s*)?id\b"      # "id", "student id", "UCI ID"
        r"(?:\s*(?:number|num|no\.?|#))?"                # optional "number" / "#"
        r"\s*(?:is|:|=|#|-)?\s*"                         # optional "is" / ":" / "="
        r"(\d{8})\b"                                     # the 8 digits (only this part is masked)
    )

    def __init__(self):
        super().__init__(supported_entities=["UCI_ID"], name="uci_id_context")

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        return [RecognizerResult("UCI_ID", m.start(1), m.end(1), 0.85)
                for m in self.PATTERN.finditer(text)]


GLINER_MODEL = "urchade/gliner_multi_pii-v1"   # PII-tuned GLiNER (~1.2 GB, downloaded once)
GLINER_LABELS = ["person"]
GLINER_THRESHOLD = 0.5
GLINER_CHUNK_CHARS = 1000                     # GLiNER reads ~384 tokens at a time


class GlinerPersonRecognizer(EntityRecognizer):
    """Alternative to spaCy for finding names: a GLiNER model, asked for "person" spans."""

    def __init__(self):
        self._model = None
        super().__init__(supported_entities=["PERSON"], name="gliner_person")

    def load(self):
        pass  # the model is loaded on first use, so `--ner spacy` never pays for it

    def _get_model(self):
        if self._model is None:
            import os
            import warnings
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            warnings.filterwarnings("ignore")
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(GLINER_MODEL)
        return self._model

    @staticmethod
    def _chunks(text):
        """Yield (offset, piece) pieces of at most GLINER_CHUNK_CHARS, split on line ends."""
        start = 0
        while start < len(text):
            end = min(start + GLINER_CHUNK_CHARS, len(text))
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl + 1
            yield start, text[start:end]
            start = end

    def analyze(self, text, entities, nlp_artifacts=None):
        model = self._get_model()
        results = []
        for offset, piece in self._chunks(text):
            for e in model.predict_entities(piece, GLINER_LABELS, threshold=GLINER_THRESHOLD):
                results.append(RecognizerResult("PERSON", offset + e["start"], offset + e["end"],
                                                float(e["score"])))
        return results


def name_in_code_regex(words):
    """Roster surnames / UCInetIDs inside code identifiers (nakamuraSolution, jnakamu3_hw3.py)."""
    parts = []
    for w in sorted(words, key=len, reverse=True):
        lo, cap, up = re.escape(w.lower()), re.escape(w.capitalize()), re.escape(w.upper())
        parts += [
            rf"(?<![A-Za-z]){lo}(?![a-z])",   # nakamuraSolution, nakamura_hw3
            rf"{cap}(?![a-z])",               # solveNakamura, MyNakamuraClass
            rf"(?<![A-Z]){up}(?![A-Za-z])",   # NAKAMURA_CONST
        ]
    return "|".join(parts)


analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(PatternRecognizer(
    supported_entity="USERNAME",
    name="home_dir_username",
    patterns=[Pattern(name="home_dir_username", regex=HOME_DIR_USERNAME, score=0.95),
              Pattern(name="profile_url_username", regex=PROFILE_URL_USERNAME, score=0.95)],
))
analyzer.registry.add_recognizer(UciIdContextRecognizer())
anonymizer = AnonymizerEngine()

# Which statistical model finds names (PERSON): "spacy" (default) or "gliner".
_SPACY_RECOGNIZER = next(r for r in analyzer.registry.recognizers if r.name == "SpacyRecognizer")
_gliner_recognizer = None


def set_ner(kind="spacy"):
    global _gliner_recognizer
    analyzer.registry.remove_recognizer("SpacyRecognizer")
    analyzer.registry.remove_recognizer("gliner_person")
    if kind == "spacy":
        analyzer.registry.add_recognizer(_SPACY_RECOGNIZER)
    elif kind == "gliner":
        if _gliner_recognizer is None:
            _gliner_recognizer = GlinerPersonRecognizer()
        analyzer.registry.add_recognizer(_gliner_recognizer)
    else:
        raise ValueError(f"Unknown NER model: {kind!r} (use 'spacy' or 'gliner')")

ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS",
            "USERNAME", "STUDENT_NAME", "UCI_NETID", "UCI_ID", "NAME_IN_CODE"]

# Recognizers that depend on the roster / harvested IDs (rebuilt by configure()).
_ROSTER_RECOGNIZERS = ["roster_names", "roster_netids", "roster_uci_ids", "roster_names_in_code"]
roster_lookup = {}


def configure(use_roster=True, extra_netids=(), persist=True, roster_path=ROSTER_PATH):
    """(Re)build the roster-based recognizers. Safe to call more than once.
    persist=False: don't write new students to student_map.csv (used by tests and eval.py)."""
    global roster_lookup
    rows = read_roster_rows(roster_path) if use_roster else []
    names, netids, uci_ids, roster_lookup, common_word_ids = build_index(rows, extra_netids, persist)

    for name in _ROSTER_RECOGNIZERS:
        analyzer.registry.remove_recognizer(name)

    # Presidio refuses empty deny lists, so only add the ones that have entries.
    if names:
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="STUDENT_NAME", name="roster_names", deny_list=names))
    if netids:
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="UCI_NETID", name="roster_netids", deny_list=netids))
    if uci_ids:
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="UCI_ID", name="roster_uci_ids", deny_list=uci_ids))
    code_words = [n for n in names if " " not in n] + netids  # surnames + UCInetIDs
    if code_words:
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="NAME_IN_CODE",
            name="roster_names_in_code",
            patterns=[Pattern("roster_names_in_code", name_in_code_regex(code_words), 0.9)],
            global_regex_flags=re.DOTALL | re.MULTILINE,  # NOT ignore-case: capitalization matters
        ))
    return {"roster_students": len(rows), "netids_from_data": len(extra_netids),
            "common_word_ids": common_word_ids}


configure()  # default: use roster.csv if it exists


# ---------------------------------------------------------------------------
# 3. Placeholders (anonymization)
# ---------------------------------------------------------------------------
_non_roster_labels = {}
_non_roster_counts = defaultdict(int)


def make_operator(entity):
    def label(value: str) -> str:
        key = strip_accents(value.strip()).lower()
        matches = roster_lookup.get(key)
        if matches:
            if len({num for num, _ in matches}) > 1:
                return "<STUDENT_AMBIGUOUS>"
            num, kind = next(iter(matches))
            return f"<STUDENT_{num}>" if kind == "NAME" else f"<STUDENT_{num}_{kind}>"
        if (entity, key) not in _non_roster_labels:
            _non_roster_counts[entity] += 1
            _non_roster_labels[(entity, key)] = f"<{entity}_{_non_roster_counts[entity]}>"
        return _non_roster_labels[(entity, key)]
    return label


OPERATORS = {e: OperatorConfig("custom", {"lambda": make_operator(e)}) for e in ENTITIES}


# A one-word PERSON match is treated as a code identifier (not a name) when it is used like code:
#   user.name / repo.owner   pd.read_csv   dijkstra(   student = ...   def/class/for/import/as X
_CODE_AFTER = re.compile(r"\(|\.[A-Za-z_]|\s*=(?!=)")
_CODE_BEFORE = re.compile(r"(?:\.|\b(?:def|class|for|import|as|in|return|new|self)\s+)$")


def clean_model_names(text, results):
    """Tidy up PERSON matches from the name *models* (spaCy / GLiNER) before masking.
    Roster and regex matches are never changed.

    1. A model "name" never crosses a line break. spaCy sometimes tags
       "praghuna\\ndef average(nums" as one name, which would wipe out code.
    2. Where a model name overlaps a rule/roster match, the rule wins (it is exact,
       and keeps the <STUDENT_n> link).
    3. One-word names used like code (user.name, class Person, pd.read_csv) are dropped.
    """
    for r in results:
        if r.entity_type == "PERSON":
            span = text[r.start:r.end]
            if "\n" in span:
                first = span.split("\n", 1)[0].rstrip()
                r.end = r.start + len(first)
    results = [r for r in results if r.end > r.start]

    rule_spans = [(r.start, r.end) for r in results if r.entity_type != "PERSON"]
    results = [r for r in results
               if r.entity_type != "PERSON"
               or not any(s < r.end and r.start < e for s, e in rule_spans)]
    return _drop_code_identifiers(text, results)


def _drop_code_identifiers(text, results):
    def is_code(r):
        span = text[r.start:r.end]
        if len(span) < 2:                       # "I", stray letters
            return True
        if any(c.isspace() for c in span):      # multi-word spans look like real names
            return False
        return bool(_CODE_AFTER.match(text, r.end) or _CODE_BEFORE.search(text, 0, r.start))

    persons = [r for r in results if r.entity_type == "PERSON"]
    code_words = {text[r.start:r.end].lower() for r in persons if is_code(r)}
    # If a word is used as code anywhere in this text, drop all its PERSON matches
    # (e.g. `author = input("Enter author name")`).
    return [r for r in results
            if not (r.entity_type == "PERSON"
                    and (is_code(r) or text[r.start:r.end].lower() in code_words))]


def deidentify_with_counts(text: str):
    results = analyzer.analyze(text=text, language="en", entities=ENTITIES)
    results = clean_model_names(text, results)
    res = anonymizer.anonymize(text=text, analyzer_results=results, operators=OPERATORS)
    return res.text, [item.entity_type for item in res.items]


def deidentify(text: str) -> str:
    return deidentify_with_counts(text)[0]


# ---------------------------------------------------------------------------
# 4. CSV processing
# ---------------------------------------------------------------------------
def deidentify_csv(input_path, columns=None, use_llm=False, use_roster=True, roster_path=None):
    input_path = Path(input_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{input_path.stem}_deid.csv"
    review_path = OUTPUT_DIR / f"{input_path.stem}_llm_review.csv"  # PRIVATE: contains original text
    counts, rows_with_pii, total = Counter(), 0, 0
    review_rows = []

    # Pass 1: read the file, check columns, and harvest UCInetIDs from @uci.edu emails.
    with open(input_path, newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        cols = columns or fieldnames
        missing = [c for c in cols if c not in fieldnames]
        if missing:
            raise SystemExit(f"Columns not in file: {missing}. Available: {fieldnames}")
        netids = harvest_netids(row[c] for row in reader for c in cols)
    info = configure(use_roster=use_roster, extra_netids=netids,
                     roster_path=roster_path or ROSTER_PATH)
    print(f"Roster students: {info['roster_students']}, UCInetIDs found in data: {info['netids_from_data']}")
    if info["common_word_ids"]:
        print(f"Note: {info['common_word_ids']} surname(s)/UCInetID(s) are common words or code "
              f"keywords; they are only matched as part of a full name or email.")

    if use_llm:
        check_ollama()
        llm_label = make_operator("LLM_PII")

    # Pass 2: de-identify.
    with open(input_path, newline="", encoding="utf-8-sig") as fin, \
            open(out_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for total, row in enumerate(reader, start=1):
            found_any = False
            for c in cols:
                if not row[c]:
                    continue
                row[c], found = deidentify_with_counts(row[c])        # layer 1: rules
                if use_llm:                                           # layer 2: LLM
                    findings = llm_find_pii(row[c])
                    row[c], applied = mask_findings(row[c], findings, llm_label)
                    found += ["LLM_PII"] * len(applied)
                    for f in applied:
                        review_rows.append({"row": total, "column": c, "flagged_text": f["text"],
                                            "type": f["type"], "reason": f["reason"]})
                counts.update(found)
                found_any = found_any or bool(found)
            rows_with_pii += found_any
            writer.writerow(row)
            if total % (10 if use_llm else 100) == 0:
                print(f"  ...{total} rows")

    if review_rows:
        with open(review_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["row", "column", "flagged_text", "type", "reason"])
            w.writeheader()
            w.writerows(review_rows)

    print(f"\nDone: {total} rows -> {out_path}")
    print(f"Rows with PII masked: {rows_with_pii}")
    for entity, n in counts.most_common():
        print(f"  {entity:15} {n}")
    if use_llm:
        print(f"LLM flagged {len(review_rows)} item(s)"
              + (f" -> review them in {review_path}" if review_rows else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De-identify text columns in a CSV.")
    parser.add_argument("input", help="path to the input CSV")
    parser.add_argument("--columns", nargs="+",
                        help="columns to scan (default: all columns)")
    parser.add_argument("--llm", action="store_true",
                        help="also run the local LLM second pass (slower)")
    parser.add_argument("--no-roster", action="store_true",
                        help="ignore roster.csv (test how the pipeline does without one)")
    parser.add_argument("--ner", choices=["spacy", "gliner"], default="spacy",
                        help="model used to find names (default: spacy)")
    args = parser.parse_args()
    set_ner(args.ner)
    deidentify_csv(args.input, args.columns, use_llm=args.llm, use_roster=not args.no_roster)
