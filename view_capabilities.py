#!/usr/bin/env python3
"""
Utility script to view and manage stored capabilities.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from layers.capability_store import CapabilityStore


def view_all_capabilities(store: CapabilityStore, sort_by: str = "created_at"):
    """Display all stored capabilities."""
    capabilities = store.list_all_capabilities(sort_by=sort_by)

    if not capabilities:
        print("No capabilities stored yet.")
        return

    print("=" * 80)
    print(f"STORED CAPABILITIES ({len(capabilities)} total)")
    print("=" * 80)
    print()

    for i, cap in enumerate(capabilities, 1):
        print(f"{i}. {cap['cap_id']}")
        print(f"   Request: {cap['request']}")
        print(f"   Created: {cap['created_at']}")
        print(f"   Success: {cap['success']}")
        print(f"   Execution Time: {cap['execution_time']:.2f}s")
        print(f"   Reuse Count: {cap['reuse_count']}")
        print(f"   Last Used: {cap.get('last_used', 'Never')}")
        print(f"   Imports: {', '.join(cap['imports'])}")
        print()


def view_capability_code(store: CapabilityStore, cap_id: str):
    """Display the code for a specific capability."""
    try:
        # Load the code
        code_path = store.capabilities_dir / f"{cap_id}.py"
        metadata_path = store.capabilities_dir / f"{cap_id}.json"

        if not code_path.exists():
            print(f"Capability {cap_id} not found!")
            return

        code = code_path.read_text()
        metadata = json.loads(metadata_path.read_text())

        print("=" * 80)
        print(f"CAPABILITY: {cap_id}")
        print("=" * 80)
        print()
        print(f"Request: {metadata['request']}")
        print(f"Created: {metadata['created_at']}")
        print(f"Success: {metadata['success']}")
        print(f"Reuse Count: {metadata['reuse_count']}")
        print()
        print("-" * 80)
        print("CODE:")
        print("-" * 80)
        print(code)
        print()

    except Exception as e:
        print(f"Error loading capability: {e}")


def search_capabilities(store: CapabilityStore, query: str):
    """Search for capabilities similar to a query."""
    print(f"Searching for capabilities similar to: '{query}'")
    print()

    results = store.search_similar(query, threshold=0.0, top_k=5)  # Show all matches

    if not results:
        print("No capabilities found.")
        return

    print("=" * 80)
    print(f"SEARCH RESULTS ({len(results)} matches)")
    print("=" * 80)
    print()

    for i, result in enumerate(results, 1):
        cap_id = result['cap_id']
        similarity = result['similarity']
        request = result['request']

        # Color code by similarity
        if similarity >= 0.85:
            marker = "✓ WOULD REUSE"
        elif similarity >= 0.70:
            marker = "~ SIMILAR"
        else:
            marker = "- DIFFERENT"

        print(f"{i}. [{marker}] Similarity: {similarity:.3f}")
        print(f"   ID: {cap_id}")
        print(f"   Request: {request}")
        print()


def view_git_history(store: CapabilityStore, limit: int = 10):
    """Display Git commit history."""
    try:
        commits = list(store.repo.iter_commits(max_count=limit))

        if not commits:
            print("No Git history yet.")
            return

        print("=" * 80)
        print(f"GIT HISTORY (last {len(commits)} commits)")
        print("=" * 80)
        print()

        for commit in commits:
            timestamp = datetime.fromtimestamp(commit.committed_date)
            print(f"Commit: {commit.hexsha[:7]}")
            print(f"Date:   {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Message: {commit.message.strip()}")
            print()

    except Exception as e:
        print(f"Error reading Git history: {e}")


def capability_stats(store: CapabilityStore):
    """Display overall capability statistics."""
    capabilities = store.list_all_capabilities()

    if not capabilities:
        print("No capabilities stored yet.")
        return

    total = len(capabilities)
    successful = sum(1 for c in capabilities if c['success'])
    total_reuses = sum(c['reuse_count'] for c in capabilities)
    avg_execution_time = sum(c['execution_time'] for c in capabilities) / total

    # Most reused
    most_reused = max(capabilities, key=lambda x: x['reuse_count'])

    # Most recent
    most_recent = max(capabilities, key=lambda x: x['created_at'])

    print("=" * 80)
    print("CAPABILITY STORE STATISTICS")
    print("=" * 80)
    print()
    print(f"Total Capabilities: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {total - successful}")
    print(f"Total Reuses: {total_reuses}")
    print(f"Average Execution Time: {avg_execution_time:.2f}s")
    print()
    print("Most Reused:")
    print(f"  {most_reused['cap_id']}")
    print(f"  Request: {most_reused['request']}")
    print(f"  Reused: {most_reused['reuse_count']} times")
    print()
    print("Most Recent:")
    print(f"  {most_recent['cap_id']}")
    print(f"  Request: {most_recent['request']}")
    print(f"  Created: {most_recent['created_at']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View and manage stored capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_capabilities.py --list
  python view_capabilities.py --list --sort reuse_count
  python view_capabilities.py --code cap_20251009_143045_abc123
  python view_capabilities.py --search "count cells"
  python view_capabilities.py --stats
  python view_capabilities.py --history
        """
    )

    parser.add_argument("--list", action="store_true", help="List all capabilities")
    parser.add_argument("--sort", choices=["created_at", "reuse_count", "last_used"],
                        default="created_at", help="Sort order for --list")
    parser.add_argument("--code", metavar="CAP_ID", help="Show code for specific capability")
    parser.add_argument("--search", metavar="QUERY", help="Search for similar capabilities")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--history", action="store_true", help="Show Git commit history")
    parser.add_argument("--limit", type=int, default=10, help="Limit for history (default: 10)")

    args = parser.parse_args()

    # Initialize store
    store = CapabilityStore()

    # Execute requested action
    if args.list:
        view_all_capabilities(store, sort_by=args.sort)
    elif args.code:
        view_capability_code(store, args.code)
    elif args.search:
        search_capabilities(store, args.search)
    elif args.stats:
        capability_stats(store)
    elif args.history:
        view_git_history(store, limit=args.limit)
    else:
        # Default: show stats
        capability_stats(store)
        print()
        print("Run with --help for more options")


if __name__ == "__main__":
    main()
