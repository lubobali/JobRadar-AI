"""Load a profile into Lakebase.

    python scripts/seed_profile.py                 # reads profile.json
    python scripts/seed_profile.py --file other.json
    python scripts/seed_profile.py --show          # print the query, write nothing

The profile is what every job is ranked against, so it is worth being able to
see what it turns into before committing to it. `--show` prints the exact
string that gets embedded.

`profile.json` is gitignored: it carries a real name, a real email, and a career
history. `profile.example.json` is the shape, and ships.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jobradar import lakebase, matching

REQUIRED = ("email", "headline")


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Copy profile.example.json to profile.json and fill it in."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))

    missing = [field for field in REQUIRED if not profile.get(field)]
    if missing:
        raise SystemExit(f"{path} is missing: {', '.join(missing)}")
    return profile


def seed(profile: dict) -> dict:
    """Write the user, profile and skills. Returns what it wrote.

    Upserts throughout, so re-running after editing profile.json updates rather
    than duplicating - which is the normal case, since the profile is a thing
    you tune while watching what it ranks.
    """
    user = lakebase.run_query_one(
        """
        INSERT INTO users (email) VALUES (%s)
        ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
        RETURNING id, email
        """,
        (profile["email"],),
    )
    user_id = user["id"]

    lakebase.run_write(
        """
        INSERT INTO profiles (user_id, headline, summary, resume_text, target_titles)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            headline = EXCLUDED.headline,
            summary = EXCLUDED.summary,
            resume_text = EXCLUDED.resume_text,
            target_titles = EXCLUDED.target_titles,
            updated_at = now()
        """,
        (
            user_id,
            profile.get("headline"),
            profile.get("summary"),
            profile.get("resume_text"),
            profile.get("target_titles") or [],
        ),
    )

    # Replaced rather than merged. A skill removed from profile.json is a skill
    # deliberately removed, and leaving it behind would keep steering the
    # ranking toward work no longer wanted.
    lakebase.run_write("DELETE FROM skills WHERE user_id = %s", (user_id,))
    for skill in profile.get("skills") or []:
        lakebase.run_write(
            "INSERT INTO skills (user_id, skill) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (user_id, skill),
        )

    return {
        "user_id": user_id,
        "email": user["email"],
        "skills": len(profile.get("skills") or []),
        "target_titles": len(profile.get("target_titles") or []),
        "resume_chars": len(profile.get("resume_text") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="profile.json")
    parser.add_argument(
        "--show", action="store_true", help="Print the embedded query and write nothing."
    )
    arguments = parser.parse_args()

    profile = load(Path(arguments.file))
    query = matching.profile_query_text(profile)

    print("The string every job is ranked against:")
    print("-" * 72)
    print(query)
    print("-" * 72)
    print(f"{len(query)} of {matching.MAX_QUERY_CHARS} characters used")

    if arguments.show:
        return 0

    written = seed(profile)
    print()
    for key, value in written.items():
        print(f"  {key:<16} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
