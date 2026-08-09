"""The curated company watchlist — WHO we fetch, the single biggest lever on digest quality.

Garbage-in beats any downstream filter: point the pipeline at companies whose work is Lubo's lane
(fintech/payments data teams, data-platform shops, AI/LLM-infra) and off-criteria jobs become rare
at the source. Every board here was **probe-verified** to return real jobs (PLAN.md Decision Log
#8: a 200 with an empty body is a 404 in disguise) — the slug and source are confirmed, not
guessed. Companies with no public ATS board (Symmetric, Workwhile, Logicbroker, dbt Labs, …) are
deliberately absent rather than listed dead.

The scorer and the 65-point digest floor decide relevance; this list decides the candidate pool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from jobradar.fetchers import (
    adzuna,
    ashby,
    breezy,
    greenhouse,
    lever,
    remotive,
    usajobs,
    workday,
)
from jobradar.fetchers.workday import WorkdayBoard
from jobradar.models import Job

Fetcher = Callable[[], list[Job]]

_FETCHERS: dict[str, Callable[..., list[Job]]] = {
    "greenhouse": greenhouse.fetch_jobs,
    "lever": lever.fetch_jobs,
    "ashby": ashby.fetch_jobs,
}


@dataclass(frozen=True, slots=True)
class Board:
    """One probe-verified ATS board: which fetcher reads it, its slug, and the display name."""

    source: str
    slug: str
    company: str

    def to_fetcher(self) -> Fetcher:
        """Bind this board to a no-argument callable the pipeline can fetch.

        Greenhouse's board API already carries the employer name, so its `fetch_jobs` takes no
        `company` argument; Lever and Ashby do not name the company, so it is passed to them.
        """
        fetch = _FETCHERS[self.source]
        if self.source == "greenhouse":
            return partial(fetch, self.slug)
        return partial(fetch, self.slug, company=self.company)


# Curated 2026-07-16, all probe-verified live. Grouped by why each is on Lubo's list.
WATCHLIST: tuple[Board, ...] = (
    # Companies from his own application history (gold set) that have a public board.
    Board("ashby", "modelyst", "Modelyst"),
    Board("lever", "foodsmart", "Foodsmart"),
    Board("greenhouse", "nex", "Nex"),
    # Fintech / payments — his HelloPayments/ISO domain, where data & platform teams live.
    Board("ashby", "ramp", "Ramp"),
    Board("greenhouse", "brex", "Brex"),
    Board("greenhouse", "mercury", "Mercury"),
    Board("greenhouse", "marqeta", "Marqeta"),
    Board("ashby", "moderntreasury", "Modern Treasury"),
    Board("ashby", "plaid", "Plaid"),
    Board("greenhouse", "checkr", "Checkr"),
    Board("lever", "finix", "Finix"),
    Board("greenhouse", "highnote", "Highnote"),
    Board("ashby", "unit", "Unit"),
    Board("greenhouse", "lithic", "Lithic"),
    Board("greenhouse", "gusto", "Gusto"),
    Board("greenhouse", "melio", "Melio"),
    Board("greenhouse", "mesh", "Mesh Payments"),
    Board("ashby", "column", "Column"),
    Board("greenhouse", "found", "Found"),
    Board("greenhouse", "tabapay", "TabaPay"),
    Board("ashby", "astra", "Astra"),
    # Data-platform / data-engineering shops — his exact stack IS the job.
    Board("greenhouse", "databricks", "Databricks"),
    Board("ashby", "snowflake", "Snowflake"),
    Board("greenhouse", "fivetran", "Fivetran"),
    Board("ashby", "airbyte", "Airbyte"),
    Board("ashby", "astronomer", "Astronomer"),
    Board("ashby", "confluent", "Confluent"),
    Board("greenhouse", "cribl", "Cribl"),
    Board("greenhouse", "hightouch", "Hightouch"),
    Board("greenhouse", "sigmacomputing", "Sigma Computing"),
    Board("greenhouse", "starburst", "Starburst"),
    Board("ashby", "montecarlodata", "Monte Carlo"),
    Board("ashby", "prefect", "Prefect"),
    Board("greenhouse", "datadog", "Datadog"),
    Board("greenhouse", "stripe", "Stripe"),
    # AI / LLM infrastructure — his LLM/RAG/agent lane.
    Board("greenhouse", "anthropic", "Anthropic"),
    Board("ashby", "cohere", "Cohere"),
    Board("ashby", "langchain", "LangChain"),
    Board("ashby", "baseten", "Baseten"),
    Board("ashby", "modal", "Modal"),
    Board("ashby", "pinecone", "Pinecone"),
    Board("ashby", "weaviate", "Weaviate"),
    Board("greenhouse", "scaleai", "Scale AI"),
    # Growth batch 2026-07-20 — 60 more in-lane companies, each probe-verified live (source is the
    # board that returned real jobs). "Catch more, miss nothing": more of his targets on the ATS
    # platforms already covered, rather than new low-yield platforms.
    # Fintech / payments.
    Board("greenhouse", "adyen", "Adyen"),
    Board("greenhouse", "affirm", "Affirm"),
    Board("greenhouse", "alloy", "Alloy"),
    Board("lever", "anchorage", "Anchorage Digital"),
    Board("greenhouse", "billcom", "Bill.com"),
    Board("greenhouse", "chime", "Chime"),
    Board("ashby", "circle", "Circle"),
    Board("lever", "dwolla", "Dwolla"),
    Board("greenhouse", "fireblocks", "Fireblocks"),
    Board("greenhouse", "galileo", "Galileo"),
    Board("greenhouse", "gemini", "Gemini"),
    Board("greenhouse", "justworks", "Justworks"),
    Board("ashby", "middesk", "Middesk"),
    Board("lever", "nium", "Nium"),
    Board("ashby", "novo", "Novo"),
    Board("ashby", "paxos", "Paxos"),
    Board("greenhouse", "payoneer", "Payoneer"),
    Board("ashby", "persona", "Persona"),
    Board("ashby", "sardine", "Sardine"),
    Board("ashby", "synctera", "Synctera"),
    Board("lever", "truv", "Truv"),
    # Data platform / data engineering.
    Board("ashby", "anomalo", "Anomalo"),
    Board("ashby", "atlan", "Atlan"),
    Board("greenhouse", "clickhouse", "ClickHouse"),
    Board("ashby", "datafold", "Datafold"),
    Board("ashby", "deepnote", "Deepnote"),
    Board("greenhouse", "dremio", "Dremio"),
    Board("greenhouse", "imply", "Imply"),
    Board("ashby", "lightdash", "Lightdash"),
    Board("ashby", "materialize", "Materialize"),
    Board("ashby", "motherduck", "MotherDuck"),
    Board("ashby", "sifflet", "Sifflet"),
    Board("greenhouse", "singlestore", "SingleStore"),
    Board("lever", "tinybird", "Tinybird"),
    # AI / LLM infrastructure.
    Board("greenhouse", "amplitude", "Amplitude"),
    Board("ashby", "anyscale", "Anyscale"),
    Board("ashby", "cerebras", "Cerebras"),
    Board("ashby", "chalk", "Chalk"),
    Board("ashby", "cognition", "Cognition"),
    Board("greenhouse", "cresta", "Cresta"),
    Board("ashby", "decagon", "Decagon"),
    Board("ashby", "dust", "Dust"),
    Board("greenhouse", "fireworksai", "Fireworks AI"),
    Board("ashby", "harvey", "Harvey"),
    Board("ashby", "inngest", "Inngest"),
    Board("ashby", "lancedb", "LanceDB"),
    Board("greenhouse", "launchdarkly", "LaunchDarkly"),
    Board("ashby", "llamaindex", "LlamaIndex"),
    Board("greenhouse", "mixpanel", "Mixpanel"),
    Board("ashby", "nomic", "Nomic"),
    Board("ashby", "openai", "OpenAI"),
    Board("ashby", "perplexity", "Perplexity"),
    Board("ashby", "poolside", "Poolside"),
    Board("ashby", "posthog", "PostHog"),
    Board("ashby", "sierra", "Sierra"),
    Board("ashby", "temporal", "Temporal"),
    Board("ashby", "unstructured", "Unstructured"),
    Board("greenhouse", "vectara", "Vectara"),
    Board("ashby", "writer", "Writer"),
    Board("lever", "zilliz", "Zilliz"),
)


def to_fetchers(boards: Sequence[Board] = WATCHLIST) -> list[Fetcher]:
    """Turn a watchlist into the no-argument fetchers `run_pipeline` consumes.

    Each fetcher isolates its own failures inside the pipeline, so one dead board never sinks a run.
    """
    return [board.to_fetcher() for board in boards]


# Remotive is keyword search, not a company board, so it is driven by queries rather than slugs.
# Every result is remote → it feeds the workable digest directly ("catch more, miss nothing").
# Kept focused: broad enough to cover his lane, few enough that overlap/cost stay small.
REMOTIVE_SEARCHES: tuple[str, ...] = (
    "data engineer",
    "AI engineer",
    "machine learning engineer",
    "data platform",
    "backend engineer",
)


def remotive_fetchers(searches: Sequence[str] = REMOTIVE_SEARCHES) -> list[Fetcher]:
    """Build a Remotive fetcher per search term."""
    return [partial(remotive.fetch_jobs, term) for term in searches]


# Adzuna geocodes every posting to a physical city, so its unique value is LOCAL search in the
# metros he can work — which no ATS board can do (they are per-company, not per-place). We point it
# ONLY at his workable metros (Tampa/Sarasota target + Chicago bridge); remote is Remotive's job.
# Probe-confirmed: a Sarasota search returns Tampa TARGET_METRO jobs; a Chicago search Chicagoland.
_ADZUNA_LOCATIONS: tuple[tuple[str, int], ...] = (
    ("Sarasota", 60),  # covers Venice, Tampa, Bradenton, St. Petersburg — his target metro
    ("Chicago", 40),  # covers Chicagoland — his bridge
)
_ADZUNA_PHRASES: tuple[str, ...] = (
    "data engineer",
    "AI engineer",
    "machine learning engineer",
    "data platform",
)


@dataclass(frozen=True, slots=True)
class AdzunaQuery:
    """One Adzuna local search: a phrase within `distance` miles of `where`."""

    phrase: str
    where: str
    distance: int

    def to_fetcher(self) -> Fetcher:
        return partial(
            adzuna.fetch_jobs,
            self.phrase,
            where=self.where,
            distance=self.distance,
            max_days_old=30,
            max_results=25,
        )


ADZUNA_QUERIES: tuple[AdzunaQuery, ...] = tuple(
    AdzunaQuery(phrase, where, dist)
    for where, dist in _ADZUNA_LOCATIONS
    for phrase in _ADZUNA_PHRASES
)


def adzuna_fetchers(queries: Sequence[AdzunaQuery] = ADZUNA_QUERIES) -> list[Fetcher]:
    """Build an Adzuna fetcher per local query. Needs ADZUNA_APP_ID/APP_KEY in the environment."""
    return [q.to_fetcher() for q in queries]


# Workday = gov/defense contractors, his US-citizen/clearance MOAT — a category no other source
# covers. Boards are per-tenant with an unguessable `site` segment, so they were discovered ONCE
# (via robots.txt) and their coordinates hardcoded here rather than re-discovered every run.
# Kept lean: Workday hydrates each job in a second call, and most gov roles are onsite at a facility
# (OTHER_US, filtered from the digest). The payoff is the rare REMOTE clearance role — moat AND
# workable — plus the onsite ones staying in the full ranking for inspection.
_WORKDAY_BOARDS: tuple[tuple[WorkdayBoard, str], ...] = (
    (WorkdayBoard("gdit", "External_Career_Site", "gdit.wd5.myworkdayjobs.com"), "GDIT"),
    (WorkdayBoard("leidos", "External", "leidos.wd5.myworkdayjobs.com"), "Leidos"),
    (WorkdayBoard("parsons", "Search", "parsons.wd5.myworkdayjobs.com"), "Parsons"),
)
WORKDAY_SEARCHES: tuple[str, ...] = ("data engineer", "AI engineer")


def workday_fetchers() -> list[Fetcher]:
    """Build a Workday fetcher per (gov/defense board x search). max_results is small on purpose —
    each hit costs a second hydration call, and the yield to the workable digest is a few remote
    clearance roles."""
    return [
        partial(workday.fetch_jobs, board, term, company=company, max_results=10)
        for board, company in _WORKDAY_BOARDS
        for term in WORKDAY_SEARCHES
    ]


# USAJobs = the federal government's official API, his citizenship MOAT and a category nothing else
# covers. Every posting is federal, so — like Adzuna — it is targeted at his WORKABLE scopes (his
# metros + remote), because most federal jobs are onsite at a facility and would be filtered out.
_USAJOBS_KEYWORDS: tuple[str, ...] = ("data engineer", "artificial intelligence")
_USAJOBS_SCOPES: tuple[tuple[str | None, int | None, bool], ...] = (
    ("Chicago, Illinois", 40, False),  # his bridge
    ("Tampa, Florida", 60, False),  # his target metro (covers Sarasota/Venice)
    (None, None, True),  # nationwide, remote-only
)


@dataclass(frozen=True, slots=True)
class UsaJobsQuery:
    """One USAJobs search: a keyword, optionally scoped to a metro or to remote-only."""

    keyword: str
    location_name: str | None
    radius: int | None
    remote_only: bool

    def to_fetcher(self) -> Fetcher:
        return partial(
            usajobs.fetch_jobs,
            self.keyword,
            location_name=self.location_name,
            radius=self.radius,
            remote_only=self.remote_only,
            max_results=25,
        )


USAJOBS_QUERIES: tuple[UsaJobsQuery, ...] = tuple(
    UsaJobsQuery(keyword=kw, location_name=loc, radius=rad, remote_only=rem)
    for kw in _USAJOBS_KEYWORDS
    for (loc, rad, rem) in _USAJOBS_SCOPES
)


def usajobs_fetchers(queries: Sequence[UsaJobsQuery] = USAJOBS_QUERIES) -> list[Fetcher]:
    """Build a USAJobs fetcher per query. Needs USAJOBS_API_KEY/USAJOBS_EMAIL in the environment."""
    return [q.to_fetcher() for q in queries]


# Breezy HR — Dave Asprey's biohacking portfolio (a real interest of his, not a generic add). The
# Asprey Group runs ONE central Breezy board for all its brands (Upgrade Labs, Danger Coffee,
# TrueDark, 40 Years of Zen, Homebiotic...), so a single board covers the lot. Usually empty; wired
# in so a future engineering/e-commerce role reaches the digest the moment it posts. Discovery-
# style: the list API carries metadata only, no JD — scored on that, full posting on click-through.
_BREEZY_BOARDS: tuple[tuple[str, str], ...] = (
    ("the-asprey-group", "Dave Asprey (The Asprey Group)"),
)


def breezy_fetchers(boards: Sequence[tuple[str, str]] = _BREEZY_BOARDS) -> list[Fetcher]:
    """Build a Breezy fetcher per (subdomain, display-name) board."""
    return [partial(breezy.fetch_jobs, slug, company=name) for slug, name in boards]


def all_sources() -> list[Fetcher]:
    """Every source a full run pulls: ATS watchlist + Remotive (remote) + Adzuna (local metros) +
    Workday (gov/defense moat) + USAJobs (federal moat) + Breezy (Asprey portfolio)."""
    return (
        to_fetchers()
        + remotive_fetchers()
        + adzuna_fetchers()
        + workday_fetchers()
        + usajobs_fetchers()
        + breezy_fetchers()
    )
