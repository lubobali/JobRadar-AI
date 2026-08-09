"""Call a running JobRadar-AI MCP server the way an agent does.

Not a unit test. The suite mocks every network boundary, which is what makes it
fast and what makes it prove logic - but a mocked boundary cannot tell you that
the server binds, that the transport negotiates, that the tools are discoverable
over the wire, or that api.weather.gov likes your User-Agent today.

That gap is where the interesting failures live. On the last project every unit
test passed while the first live query died on a SQL syntax error, because a
fake cursor records SQL, it does not parse it.

Usage:
    python scripts/smoke_test.py                      # http://127.0.0.1:8000/mcp
    python scripts/smoke_test.py --url https://<app>.databricksapps.com/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from fastmcp import Client

CASES: list[tuple[str, dict]] = [
    # Reads
    ("get_profile", {}),
    ("search_jobs", {"query": "senior data engineer building Spark pipelines", "top_k": 5}),
    ("search_jobs", {"query": "remote machine learning platform", "top_k": 3,
                     "remote_only": True}),
    ("list_applications", {}),
    # The failure paths matter as much as the happy ones. An agent's honesty
    # depends on these coming back as readable errors rather than as crashes,
    # and on the write tools refusing what they should refuse.
    ("search_jobs", {"query": "   "}),
    ("get_job", {"job_id": "no-such-job"}),
    ("log_application", {"job_id": "x", "status": "in progress"}),
    ("update_application_status", {"application_id": 999999, "status": "offer"}),
    ("add_interview_note", {"application_id": "the second one", "note": "hi"}),
]


def unwrap(result: object) -> dict:
    """Pull the tool's dict out of whatever the client wrapped it in."""
    if getattr(result, "structured_content", None):
        payload = result.structured_content
        # FastMCP wraps a bare return value under "result".
        return payload.get("result", payload) if isinstance(payload, dict) else payload
    if getattr(result, "content", None):
        try:
            return json.loads(result.content[0].text)
        except Exception:
            return {"raw": result.content[0].text}
    return {}


async def run(url: str, token: str | None = None) -> int:
    failures = 0

    # The deployment behind a bearer token needs the header; the Databricks App
    # deployment needs no header at all, because the platform authenticates the
    # request before it arrives. One script, both targets.
    target = url
    if token:
        from fastmcp.client.transports import StreamableHttpTransport  # noqa: PLC0415

        target = StreamableHttpTransport(
            url=url, headers={"Authorization": f"Bearer {token}"}
        )

    async with Client(target) as client:
        tools = await client.list_tools()
        print(f"Connected to {url}")
        print(f"{len(tools)} tools discovered:\n")
        for tool in sorted(tools, key=lambda item: item.name):
            summary = (tool.description or "").strip().splitlines()[0]
            print(f"  {tool.name:<28} {summary}")
        print()
        print("-" * 78)

        for name, arguments in CASES:
            label = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
            print(f"\n{name}({label})")
            try:
                payload = unwrap(await client.call_tool(name, arguments))
            except Exception as exc:
                # A raise here is the failure this script exists to catch: the
                # tools are supposed to return errors, never throw them.
                print(f"  RAISED: {type(exc).__name__}: {exc}")
                failures += 1
                continue

            if "error" in payload:
                print(f"  error [{payload.get('error_type')}]: {payload['error']}")
                continue

            for key in ("count", "headline", "status_filter"):
                if key in payload:
                    print(f"  {key}: {payload[key]}")
            for job in (payload.get("results") or [])[:5]:
                score = job.get("fit_score")
                print(
                    f"  {job.get('similarity', 0):.3f}  "
                    f"{(job.get('title') or '')[:38]:<38} "
                    f"{(job.get('company') or '')[:22]:<22} "
                    f"{'score ' + str(score) if score is not None else ''}"
                )
            for app in (payload.get("applications") or [])[:5]:
                print(f"  [{app['status']}] {app['title']} at {app['company']}")


    print("\n" + "-" * 78)
    print("every tool returned a result" if not failures else f"{failures} tool(s) RAISED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument(
        "--token",
        default=os.environ.get("JOBRADAR_BEARER_TOKEN"),
        help="Bearer token, if the target requires one. Defaults to $JOBRADAR_BEARER_TOKEN.",
    )
    arguments = parser.parse_args()
    return asyncio.run(run(arguments.url, arguments.token))


if __name__ == "__main__":
    sys.exit(main())
