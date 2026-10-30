#!/usr/bin/env python
"""Thin wrapper so the tool can be run as `python scripts/train_tokenizer.py`."""

from biln_lm.cli.train_tokenizer import main

if __name__ == "__main__":
    main()
