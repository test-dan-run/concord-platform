// Both apps link to `/` and `/models/:id`, so those must keep opening the Model Catalog's pages. The pages
// are stubbed: this checks the routing and the Nav's catalog switch, not what the pages fetch.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { catalogOf } from "./Nav";
import { routes } from "./routes";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));
vi.mock("../lib/api", () => ({ api: { get: () => new Promise(() => {}) } }));
vi.mock("../pages/ModelsPage", () => ({ ModelsPage: () => <h1>models page</h1> }));
vi.mock("../pages/ModelDetail", () => ({ ModelDetail: () => <h1>model detail page</h1> }));
vi.mock("../pages/DatasetsPage", () => ({ DatasetsPage: () => <h1>datasets page</h1> }));

function open(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

const tab = () => screen.getByRole("group", { name: "Catalog" }).querySelector('[aria-pressed="true"]')?.textContent;

afterEach(cleanup);

describe("routes", () => {
  it("keeps / and /models/:id on the Model Catalog's pages", () => {
    open("/");
    expect(screen.getByRole("heading", { name: "models page" })).toBeTruthy();
    expect(tab()).toBe("Model Catalog");
    cleanup();
    open("/models/whisper-large-v3");
    expect(screen.getByRole("heading", { name: "model detail page" })).toBeTruthy();
    expect(tab()).toBe("Model Catalog");
  });

  it("opens the Dataset Catalog at /datasets", () => {
    open("/datasets");
    expect(screen.getByRole("heading", { name: "datasets page" })).toBeTruthy();
    expect(tab()).toBe("Dataset Catalog");
    expect(screen.queryByRole("link", { name: "Add model" })).toBeNull();
  });

  it("switches catalogs from the Nav", () => {
    const router = open("/models/whisper-large-v3");
    fireEvent.click(screen.getByRole("button", { name: "Dataset Catalog" }));
    expect(router.state.location.pathname).toBe("/datasets");
    fireEvent.click(screen.getByRole("button", { name: "Model Catalog" }));
    expect(router.state.location.pathname).toBe("/");
  });

  it("puts everything under /datasets in the Dataset Catalog", () => {
    expect(catalogOf("/")).toBe("models");
    expect(catalogOf("/models/x")).toBe("models");
    expect(catalogOf("/datasets")).toBe("datasets");
    expect(catalogOf("/datasets/abc/v/2")).toBe("datasets");
    expect(catalogOf("/datasetsx")).toBe("models");
  });
});
