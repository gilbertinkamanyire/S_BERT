# VERITAS VeriText

VERITAS VeriText is a local academic writing workbench for checking AI-written text, finding copied or rewritten source text, reviewing writing style, humanizing text, and exporting reports.

## Start the app

Windows:

```bat
scripts\run_veritext.bat
```

Or use Python:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5050.

## Optional Gemma humanizer

Copy `.env.example` to `.env`, add your own Google AI Studio key, and restart the app.
Never share or commit the key.

## Tests

```powershell
python -m unittest discover -s tests -p "test*.py"
```

## Project layout

```text
backend/               Backend Python application
  app.py               Server launcher
  veritext_app.py      Flask routes and APIs
  veritext_core.py     Writing and plagiarism analysis
  engine.py            Local text vectors and metrics
  detector.py          AI-writing classifier
  ingest.py            PDF, Word, and text input
frontend/              Browser interface
  templates/           Main HTML page and function-based partials
  templates/partials/  Header, hero, detector, results, footer, and tool sections
  static/css/          Interface styling
app.py                 Top-level launcher
tests/                 Automated tests
scripts/               Windows launch and setup tools
docs/                  Research documents
data/uploads/          Temporary uploaded files
```

The browser keeps up to 50 recent checks and humanized results on the current computer and browser. Clear them from the History screen when needed.
