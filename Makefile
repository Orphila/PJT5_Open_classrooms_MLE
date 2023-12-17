.venv:
    python3 -m venv .venv/

install: .venv
    .venv/bin/python -m pip install -r requirements.txt

clean:
    rm -rf .venv/