# conftest.py — Shared pytest fixtures and configuration

import sys
import os

# Ensure the project root is on sys.path so all modules resolve
sys.path.insert(0, os.path.dirname(__file__))
