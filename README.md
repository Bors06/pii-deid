# pii-deid

Local de-identification of student data (chat messages and code) before it is graded by an
LLM-as-a-judge. Built on [Microsoft Presidio](https://github.com/microsoft/presidio).
Everything runs on your own machine; no data is sent to an external service.

Personal information is replaced with consistent placeholders, so the judge can still read
the code and follow who did what:

```
IN:  Jordan Nakamura helped Priya Raghunathan; see nakamuraSolution() in C:\Users\praghuna\hw3
OUT: <STUDENT_1> helped <STUDENT_2>; see <STUDENT_1>Solution() in C:\Users\<STUDENT_2_NETID>\hw3
```

## How it works

Three layers, run in order:

1. **Rules** (exact, repeatable)
   - Presidio built-ins: emails, phone numbers, IP addresses
   - Roster matching (optional but very helpful): names, UCInetIDs, UCI IDs, emails. Ignores case and accents.
     Surnames that are common words or code keywords (e.g. Long) only match as full names.
   - Usernames in file paths (`C:\Users\<name>`, `/Users/<name>`, `/home/<name>`) and profile
     links (`github.com/<name>`, `linkedin.com/in/<name>`, ...)
   - UCI IDs: 8-digit numbers right after "student ID", "UCI ID", "ID:", ...
   - Roster surnames / UCInetIDs inside code identifiers (`nakamuraSolution`, `jnakamu3_hw3.py`)
   - UCInetIDs collected from `netid@uci.edu` addresses found in the data
2. **Name model**: finds names that are not on the roster. spaCy (`en_core_web_lg`) by default;
   GLiNER with `--ner gliner`. Model-detected names that are clearly code (`user.name`,
   `class Person`) or that cross a line break are discarded. spaCy is a faster model and has been
   sufficient for the tests so far.
3. **Local LLM** (optional, `--llm`): a small model via Ollama lists anything personal that
   is still visible (e.g. "the only transfer student from CityName"). It never rewrites text;
   findings are validated and logged to a review file. The model can be replaced with a larger
   one (e.g. on a lab GPU) by changing one line (`MODEL` in `llm_check.py`).

Placeholders: `<STUDENT_n>` (roster student), `<STUDENT_n_NETID>` / `_UCIID` / `_EMAIL`,
`<STUDENT_AMBIGUOUS>` (name shared by two students), `<PERSON_n>`, `<EMAIL_ADDRESS_n>`, ...
Student numbers are kept stable across runs in `student_map.csv`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Optional:
- LLM pass: install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b`
- GLiNER: `pip install gliner sentencepiece protobuf`

## Roster (optional)

Put `roster.csv` in the project folder:

```
first_name,last_name,ucinetid,uci_id
Jordan,Nakamura,jnakamu3,48213377
```

An optional `email` column is also used. Without a roster the pipeline still runs, with lower
recall on student names and IDs.

## Usage

```
python deid.py data/messages.csv --columns message_content            # rules + spaCy
python deid.py data/messages.csv --columns message_content --llm      # + LLM pass
python deid.py data/messages.csv --no-roster                          # ignore roster.csv
python deid.py data/messages.csv --ner gliner                         # GLiNER for names
```

Output goes to `output/<name>_deid.csv` (every row kept; unscanned columns unchanged).
With `--llm`, `output/<name>_llm_review.csv` lists what the LLM masked and why.

## Evaluation

A gold file is a CSV with `row_id,text,pii`, where `pii` lists the PII in that row separated by
`|` (empty = clean row).

```
python eval.py --gold examples/chat_gold.csv --no-roster
python eval.py --gold data/real_gold.csv --llm --ner both
```

Reports the catch rate (share of PII items no longer visible), clean rows that were changed,
and every leak. The sets in `examples/` are made up and only check that things work; real
performance has to be measured on hand-labeled real data.

## Tests

```
python -m pytest -q                 # all tests
python -m pytest -q -m "not gliner" # skip the GLiNER test
```

Tests use `examples/test_roster.csv` and a temporary student map, so real data is never read.
`test_llm.py` is a manual check of the LLM pass (needs Ollama running).

## Privacy

Never commit `roster.csv`, `student_map.csv` (links placeholders back to students), `data/`,
or `output/` (the LLM review file contains original text). They are in `.gitignore`.

## Known limitations

- Nicknames / preferred names not on the roster, and handles that differ from the UCInetID
- spaCy sometimes tags capitalized words or famous names as people ("Mark the task", "Dijkstra")
- Indirect identifiers (hometown, "only X in my class") are only partly caught, by the LLM
- LLM pass on CPU is slow (~4 s per message); use a GPU and a larger model for big datasets
