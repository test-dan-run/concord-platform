"""One test per (service, check); parametrized in conftest.py from the endpoints map."""


def test_conformance(live_client, svc, check):
    check(live_client, svc)
