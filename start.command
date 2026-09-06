#!/bin/zsh
cd -- "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
  echo '初回セットアップが必要です。README.md を参照してください。'
  read -k 1
  exit 1
fi
exec .venv/bin/python app.py
