#!/bin/bash
.venv/bin/python3 -m ensurepip
.venv/bin/python3 -m pip install --no-index mamba-ssm==2.2.4 || .venv/bin/python3 -m pip install mamba-ssm==2.2.4
