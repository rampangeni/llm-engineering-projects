"""Prompts parameterised by language."""

from __future__ import annotations

import json

LANGUAGES = {
    "C++": {"ext": "cpp", "fence": "cpp", "filename": "main.cpp"},
    "Rust": {"ext": "rs", "fence": "rust", "filename": "main.rs"},
}


def system_prompt(language: str) -> str:
    return f"""
Your task is to convert Python code into high performance {language} code.
Respond only with {language} code. Do not provide any explanation other than occasional comments.
The {language} response needs to produce an identical output in the fastest possible time.
"""


def user_prompt(python: str, language: str, compile_command: list[str], system_info: dict | None = None) -> str:
    sysinfo = json.dumps(system_info, indent=1) if system_info else "(not provided)"
    filename = LANGUAGES[language]["filename"]
    return f"""
Port this Python code to {language} with the fastest possible implementation that produces identical output in the least time.
The system information is:
{sysinfo}
Your response will be written to a file called {filename} and then compiled and executed; the compilation command is:
{compile_command}
Respond only with {language} code.
Python code to port:

```python
{python}
```
"""


def messages_for(python: str, language: str, compile_command: list[str], system_info: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": user_prompt(python, language, compile_command, system_info)},
    ]
