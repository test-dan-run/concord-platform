"""Contract conformance suites.

Live:        just conform NAME          (pytest contracts/conformance --service NAME)
In-process:  from conformance import checks_for, service_meta
             for check in checks_for(meta): check(test_client, meta)
"""

from .checks import Check, checks_for
from .inprocess import load_server, running, service_meta

__all__ = ["Check", "checks_for", "load_server", "running", "service_meta"]
