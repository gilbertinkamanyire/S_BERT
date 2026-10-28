"""
VERITAS + VeriText Unified Server Launcher
=============================================================================
This file points directly to the unified VeriText app, ensuring full backward
compatibility regardless of whether the user or launcher executes `python app.py`
or `python veritext_app.py`.
"""

from .veritext_app import app
from .config import get_port

def main():
    port = get_port()
    print('=' * 75)
    print('  VERITEXT + VERITAS — Writing and Source Check')
    print('  AI writing and copied text check · Runs on your computer')
    print(f'  Server active at: http://127.0.0.1:{port}')
    print('=' * 75)
    app.run(host='127.0.0.1', port=port, debug=False)


if __name__ == '__main__':
    main()
