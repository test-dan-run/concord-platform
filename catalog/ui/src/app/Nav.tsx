import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { api, type Lease, type PlatformInfo } from "../lib/api";
import { pct } from "../lib/format";
import * as I from "../components/icons";
import { Bar, cx, Toggle } from "../components/ui";

/** The two catalogs the service shows. The tab follows the URL: everything under /datasets is the Dataset Catalog. */
export type Catalog = "models" | "datasets";

const CATALOGS: { value: Catalog; label: string; home: string }[] = [
  { value: "models", label: "Model Catalog", home: "/" },
  { value: "datasets", label: "Dataset Catalog", home: "/datasets" },
];

export function catalogOf(pathname: string): Catalog {
  return pathname === "/datasets" || pathname.startsWith("/datasets/") ? "datasets" : "models";
}

function Item({ to, icon, children, end = true }: { to: string; icon: ReactNode; children: ReactNode; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cx("flex items-center gap-2.5 px-2.5 min-h-10 rounded-md no-underline", isActive ? "bg-active !text-fg" : "!text-muted hover:!text-fg hover:bg-track")
      }
    >
      {icon}
      {children}
    </NavLink>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-muted uppercase tracking-[0.06em] px-2.5 pb-1.5">{title}</span>
      {children}
    </div>
  );
}

/** GPU memory in use against util_limit, and the queue length: the two numbers worth a glance. */
function Compute({ live }: { live: boolean }) {
  const p = useQuery({ queryKey: ["platform"], queryFn: () => api.get<PlatformInfo>("/platform"), refetchInterval: 10_000 });
  const leases = useQuery({ queryKey: ["leases"], queryFn: () => api.get<Lease[]>("/leases"), refetchInterval: 10_000 });
  const queued = (leases.data ?? []).filter((l) => l.state === "queued").length;
  const used = p.data ? p.data.used_fraction / p.data.util_limit : 0;
  return (
    <NavLink to="/queue" className="mt-auto flex flex-col gap-2 p-3 border border-track rounded-md no-underline !text-fg hover:bg-track">
      <span className="flex items-center justify-between gap-2 text-xs whitespace-nowrap">
        <span>GPU memory</span>
        <span className="num text-[11px] text-muted">{p.data ? `${(p.data.used_fraction * p.data.gpu_total_gb).toFixed(1)} / ${(p.data.util_limit * p.data.gpu_total_gb).toFixed(1)} GB` : "—"}</span>
      </span>
      <Bar value={used} color={used > 0.9 ? "var(--color-warn)" : "var(--color-primary)"} />
      <span className="flex items-center gap-2 text-[11px] text-muted">
        <span className={cx("size-2 rounded-full", queued ? "bg-warn" : "bg-subtle")} />
        {queued ? `${queued} waiting for compute` : "Queue empty"}
        <span className="ml-auto num" title={live ? "Live updates connected" : "Live updates disconnected; refreshing every 10 s"}>
          {live ? "live" : "polling"}
        </span>
      </span>
      {p.data?.platform_error && <span className="text-[11px] text-bad">Manifests invalid: using the last valid set</span>}
      <span className="sr-only">{pct(used)} of the usable GPU memory in use</span>
    </NavLink>
  );
}

/** Where the Storage widget goes while the Dataset Catalog is open (store size, audio held only, clean-up).
 *  The dataset store's API isn't served yet, so this only keeps the spot. */
function Storage() {
  return (
    <div className="mt-auto flex flex-col gap-2 p-3 border border-track rounded-md">
      <span className="text-xs">Storage</span>
      <span className="text-[11px] text-muted">Not set up yet</span>
    </div>
  );
}

export function Nav({ live }: { live: boolean }) {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const navigate = useNavigate();
  const catalog = catalogOf(loc.pathname);
  const name = CATALOGS.find((c) => c.value === catalog)!.label;
  useEffect(() => setOpen(false), [loc.pathname]);
  useEffect(() => {
    document.title = name;
  }, [name]);
  return (
    <div className="bg-panel border-b lg:border-b-0 lg:border-r border-track lg:w-[216px] lg:shrink-0">
      <nav aria-label="Primary" className="px-3 py-4 lg:py-5 flex flex-col gap-6 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto scroll-thin">
        <div className="flex items-center gap-2.5 px-2">
          <span className="text-primary">
            <I.Logo size={22} />
          </span>
          <div className="flex flex-col">
            <span className="font-semibold text-[15px] tracking-tight">{name}</span>
            <span className="text-[11px] text-muted">AI platform control plane</span>
          </div>
          <button
            type="button"
            aria-expanded={open}
            aria-controls="nav-groups"
            onClick={() => setOpen((o) => !o)}
            className="lg:hidden ml-auto min-h-10 px-3 rounded-md border border-line bg-transparent text-fg text-[13px] cursor-pointer"
          >
            {open ? "Close" : "Menu"}
          </button>
        </div>
        <Toggle
          label="Catalog"
          value={catalog}
          onChange={(v) => navigate(CATALOGS.find((c) => c.value === v)!.home)}
          options={CATALOGS}
          className="-mt-2 [&>button]:grow [&>button]:px-1.5 [&>button]:text-[11px] [&>button]:whitespace-nowrap"
        />
        <div id="nav-groups" className={cx("flex-col gap-6 flex-1", open ? "flex" : "hidden lg:flex")}>
          {catalog === "models" ? (
            <>
              <Group title="Models">
                <Item to="/" icon={<I.Cpu />} end={false}>Models</Item>
                <Item to="/add" icon={<I.Plus />}>Add model</Item>
                <Item to="/export" icon={<I.Export />}>Export</Item>
              </Group>
              <Group title="Compute">
                <Item to="/queue" icon={<I.List />}>Queue and leases</Item>
                <Item to="/index" icon={<I.Database />}>Output index</Item>
              </Group>
              <Group title="Reference">
                <Item to="/contract" icon={<I.Doc />}>Model contract</Item>
                <Item to="/docs" icon={<I.Book />}>Docs</Item>
              </Group>
              <Compute live={live} />
            </>
          ) : (
            <>
              <Group title="Datasets">
                <Item to="/datasets" icon={<I.Database />}>Datasets</Item>
              </Group>
              <Storage />
            </>
          )}
        </div>
      </nav>
    </div>
  );
}
