// The Status / Model card switch on a model's page. The tab lives in the URL (?tab=card) so a card can be linked.

import { useSearchParams } from "react-router-dom";
import { Toggle } from "./ui";

export function useDetailTab(): ["status" | "card", (t: string) => void] {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "card" ? "card" : "status";
  return [tab, (t) => setParams(t === "card" ? { tab: "card" } : {}, { replace: true })];
}

export function DetailTabs({ tab, onChange }: { tab: string; onChange: (t: string) => void }) {
  return (
    <div className="self-start">
    <Toggle
      label="Model page section"
      value={tab}
      onChange={onChange}
      options={[
        { value: "status", label: "Status" },
        { value: "card", label: "Model card" },
      ]}
    />
    </div>
  );
}
