"""MkDocs hooks for documentation build behavior."""

import logging


def on_config(config, **kwargs):
    """Reduce third-party parser warning noise under strict builds."""
    logging.getLogger("griffe").setLevel(logging.ERROR)
    return config
