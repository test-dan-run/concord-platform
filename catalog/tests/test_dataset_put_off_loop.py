"""PUT /api/store/audio does its blocking store work off the event loop, so an upload that waits for the datasets
database leaves the async routes (the gateway, the event stream) free to answer."""

import threading
import time

from catalog_helpers import sha, wav
from fastapi.testclient import TestClient

from catalog.app import create_app


def test_a_put_waiting_for_the_datasets_db_leaves_async_routes_free(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        db = app.state.datasets.db
        held = threading.Event()

        def hold() -> None:
            with db._lock:
                held.set()
                time.sleep(1.0)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait()
        d = wav(1)
        put: dict = {}

        def do_put() -> None:
            t = time.perf_counter()
            put["r"] = c.put(f"/api/store/audio/{sha(d)}", params={"offset": 0}, content=d)
            put["s"] = time.perf_counter() - t

        th = threading.Thread(target=do_put)
        th.start()
        time.sleep(0.1)  # the PUT is waiting for the database now
        t = time.perf_counter()
        r = c.get("/m/nope/x")  # an async route: the gateway's 404 for an unknown model
        took = time.perf_counter() - t
        th.join()
        holder.join()
    app.state.datasets.db.close()
    assert r.status_code == 404
    assert took < 0.3, f"the gateway waited {took:.2f} s for an upload's database lock"
    assert put["r"].json() == {"size": len(d)} and put["s"] > 0.5  # the PUT did wait, and then finished
