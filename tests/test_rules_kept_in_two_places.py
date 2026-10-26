"""Rules that are written in more than one place on purpose stay the same in each.

aicore ships to the apps without tools/, so it keeps its own copy of the base paths. aisvc ships inside custom
images without contracts/, so it keeps its own routes, and so does the catalog's verifier. These tests fail when
a copy drifts from tools/schema.py or contracts/api_types.yaml.
"""

import pytest
from aicore import endpoints
from aisvc import inspired_app, openai_app

from catalog.verifier import probes
from tools import schema

API_TYPES = schema.load_yaml(schema.PLATFORM_DIR / "contracts" / "api_types.yaml")


def contract_routes(contract: str | None) -> dict[str, str]:
    """api type -> route in contracts/api_types.yaml, for one contract or for all of them."""
    return {api: spec["route"] for c, apis in API_TYPES.items() if contract in (None, c) for api, spec in apis.items()}


def test_aicore_base_paths_are_the_platforms():
    assert endpoints.BASE_PATHS == schema.BASE_PATHS
    assert set(schema.BASE_PATHS) == set(API_TYPES)  # one base path per contract
    for contract, apis in API_TYPES.items():
        for api, spec in apis.items():
            assert spec["route"].startswith(schema.BASE_PATHS[contract] + "/"), (contract, api)


ROUTE_MAPS = {  # name -> (api type -> route, the contract it serves or None for any)
    "aisvc openai_app.ROUTES": (openai_app.ROUTES, "openai"),
    "aisvc inspired_app.INSPIRED_APIS": (inspired_app.INSPIRED_APIS, "openai-inspired"),
    "catalog verifier probes.ROUTES": (probes.ROUTES, None),
}


@pytest.mark.parametrize("name", ROUTE_MAPS)
def test_route_maps_agree_with_the_api_types(name):
    routes, contract = ROUTE_MAPS[name]
    known = contract_routes(contract)
    assert set(routes) <= set(known), f"{name} has api types that contracts/api_types.yaml does not"
    wrong = {api: (route, known[api]) for api, route in routes.items() if route != known[api]}
    assert not wrong, f"{name} routes differ from contracts/api_types.yaml (theirs, contract's): {wrong}"


def test_api_type_names_are_unique_across_contracts():
    names = [api for apis in API_TYPES.values() for api in apis]
    assert len(names) == len(set(names))  # probes.ROUTES finds a route by api type alone
