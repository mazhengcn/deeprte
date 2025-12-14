"""Stub for logging utilities. Right now just meant to avoid raw prints."""


def log(user_str: str) -> None:
    print(user_str, flush=True)
