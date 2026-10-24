import { createBrowserRouter, Navigate, Outlet, type RouteObject } from "react-router-dom";
import { useCatalogEvents } from "../lib/events";
import { AddModel } from "../pages/AddModel";
import { ContractPage } from "../pages/ContractPage";
import { DatasetsPage } from "../pages/DatasetsPage";
import { DocsPage } from "../pages/DocsPage";
import { ExportPage } from "../pages/ExportPage";
import { IndexPage } from "../pages/IndexPage";
import { ModelDetail } from "../pages/ModelDetail";
import { ModelsPage } from "../pages/ModelsPage";
import { QueuePage } from "../pages/QueuePage";
import { Nav } from "./Nav";

function Shell() {
  const live = useCatalogEvents();
  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      <Nav live={live} />
      <main className="flex-1 min-w-0 px-4 py-5 lg:px-8 lg:py-7 flex flex-col gap-5 max-w-[1440px]">
        <Outlet />
      </main>
    </div>
  );
}

// The Model Catalog's paths are fixed: both apps link to `/` and `/models/:id`. The Dataset Catalog lives under /datasets.
export const routes: RouteObject[] = [
  {
    element: <Shell />,
    children: [
      { path: "/", element: <ModelsPage /> },
      { path: "/models", element: <Navigate to="/" replace /> },
      { path: "/models/:id", element: <ModelDetail /> },
      { path: "/add", element: <AddModel /> },
      { path: "/queue", element: <QueuePage /> },
      { path: "/index", element: <IndexPage /> },
      { path: "/contract", element: <ContractPage /> },
      { path: "/docs", element: <DocsPage /> },
      { path: "/export", element: <ExportPage /> },
      { path: "/datasets", element: <DatasetsPage /> },
      { path: "/datasets/*", element: <Navigate to="/datasets" replace /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
