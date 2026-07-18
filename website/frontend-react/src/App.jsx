import PinnedActs from "./components/PinnedActs";
import Act4NewWorld from "./components/acts/Act4NewWorld";
import Act5Invitation from "./components/acts/Act5Invitation";
import Nav from "./components/layout/Nav";
import ScrollProgress from "./components/layout/ScrollProgress";

/**
 * The whole single-continuous-page structure: Acts 1-3 live inside
 * PinnedActs (one scroll-jacked section, one Canvas — see that file),
 * Acts 4-5 scroll natively beneath it. Nav/ScrollProgress are page-level
 * chrome, not owned by any one act — see their own files for how they
 * read shared scroll state without re-deriving it themselves.
 */
export default function App() {
  return (
    <>
      <Nav />
      <ScrollProgress />
      <main>
        <PinnedActs />
        <Act4NewWorld />
        <Act5Invitation />
      </main>
    </>
  );
}
