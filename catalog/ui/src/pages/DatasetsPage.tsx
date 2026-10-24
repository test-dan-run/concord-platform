// The Dataset Catalog's home: every dataset with its latest version. A placeholder until the dataset
// store's API is served (DATASET_CATALOG_PLAN.md §6).

import { Empty, PageHeader, Panel } from "../components/ui";

export function DatasetsPage() {
  return (
    <>
      <PageHeader
        crumbs="Dataset Catalog"
        title="Datasets"
        subtitle="Versioned datasets that Concord Bench and Concord Label both read. The audio is stored once and the apps never copy it."
      />
      <Panel>
        <Empty>No datasets yet. The Dataset Catalog isn't set up on this platform yet.</Empty>
      </Panel>
    </>
  );
}
