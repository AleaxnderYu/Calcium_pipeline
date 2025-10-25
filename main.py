#!/usr/bin/env python3
"""
Main entry point for the Calcium Imaging Agentic System.
Run calcium imaging analysis via command line.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import configuration and logging setup
import config
from core.logging_config import setup_logging
from graph.workflow import run_workflow
from core.session_manager import get_session_manager


def main():
    """Main entry point with CLI argument parsing."""

    parser = argparse.ArgumentParser(
        description="Calcium Imaging Agentic Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --request "Count the number of cells" --images ./data/images
  python main.py --request "Calculate mean intensity over time" --images ./data/images --verbose
  python main.py --request "Detect calcium transients" --images ./data/images --rebuild-rag
        """
    )

    # Required arguments
    parser.add_argument(
        "--request",
        type=str,
        required=True,
        help="Natural language analysis request"
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to directory containing PNG image frames"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_DIR),
        help=f"Output directory (default: {config.OUTPUT_DIR})"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--rebuild-rag",
        action="store_true",
        help="Force rebuild of RAG vector database"
    )

    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session ID for reusing E2B sandbox (creates new if not provided)"
    )

    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Force create a new session even if --session is provided"
    )

    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all active sessions and exit"
    )

    parser.add_argument(
        "--close-session",
        type=str,
        default=None,
        help="Close a specific session and exit"
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(
        log_level=config.LOG_LEVEL,
        log_file=config.LOG_FILE,
        verbose=args.verbose
    )

    # Get session manager
    session_manager = get_session_manager()

    # Handle session management commands
    if args.list_sessions:
        sessions = session_manager.list_sessions()
        print("=" * 50)
        print("ACTIVE SESSIONS")
        print("=" * 50)
        if not sessions:
            print("No active sessions")
        else:
            for session in sessions:
                print(f"\nSession: {session['session_id']}")
                print(f"  Created: {session['created_at']}")
                print(f"  Queries: {session['query_count']}")
                print(f"  Age: {session['age_seconds']:.1f}s")
                print(f"  Sandbox: {'active' if session['sandbox_active'] else 'not created yet'}")
        print("=" * 50)
        return 0

    if args.close_session:
        print(f"Closing session: {args.close_session}")
        session_manager.close_session(args.close_session)
        print("✓ Session closed")
        return 0

    # Validate required args for analysis (unless using session management commands)
    if not args.list_sessions and not args.close_session:
        if not args.request or not args.images:
            parser.error("--request and --images are required for analysis")

    # Handle session creation/selection
    if args.new_session or args.session is None:
        session_id = session_manager.create_session(args.session)
        print(f"Created new session: {session_id}\n")
    else:
        session_id = args.session
        if session_manager.get_session(session_id) is None:
            session_id = session_manager.create_session(session_id)
            print(f"Created new session: {session_id}\n")
        else:
            session = session_manager.get_session(session_id)
            print(f"Using existing session: {session_id} (queries: {session.query_count})\n")

    # Print header
    print("=" * 50)
    print("CALCIUM IMAGING ANALYSIS")
    print("=" * 50)
    print(f"Session: {session_id}")
    print(f"Request: {args.request}")
    print(f"Images: {args.images}")
    print()

    try:
        # Rebuild RAG if requested
        if args.rebuild_rag:
            print("Rebuilding RAG vector database with enhanced section-based chunking...")
            from tools.rag_system_enhanced import EnhancedRAGSystem
            rag = EnhancedRAGSystem()
            rag.rebuild()
            print("✓ RAG database rebuilt with section-based chunking\n")

        # Run workflow
        print("[Step 1/5] Preprocessing...", end=" ", flush=True)
        print("[Step 2/5] Retrieving methods...", end=" ", flush=True)
        print("[Step 3/5] Generating code...", end=" ", flush=True)
        print("[Step 4/5] Executing analysis...", end=" ", flush=True)
        print("[Step 5/5] Formatting output...", end=" ", flush=True)

        # Actually run the workflow (progress is logged, not printed)
        result = run_workflow(
            user_request=args.request,
            images_path=args.images,
            session_id=session_id
        )

        print()  # New line after progress

        # Check if analysis succeeded
        if result.metadata.get("status") == "failed":
            print("\n" + "=" * 50)
            print("ANALYSIS FAILED")
            print("=" * 50)
            print(f"Error: {result.data.get('error', 'Unknown error')}")
            print()
            return 1

        # Print results
        print("✓ Analysis completed successfully!\n")
        print("=" * 50)
        print("RESULTS")
        print("=" * 50)

        for key, value in result.data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print()

        # Save outputs
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(args.output) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save report JSON
        report_path = output_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save generated code
        code_path = output_dir / "generated_code.py"
        with open(code_path, 'w') as f:
            f.write(result.code_used)

        print("=" * 50)
        print("OUTPUT FILES")
        print("=" * 50)
        if result.figures:
            print(f"Figures: {', '.join(result.figures)}")
        print(f"Report: {report_path}")
        print(f"Code: {code_path}")
        print()

        print(f"Full output saved to: {output_dir}")
        print("=" * 50)

        return 0

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n{'=' * 50}")
        print("ERROR")
        print("=" * 50)
        print(f"{type(e).__name__}: {str(e)}")
        print()

        if args.verbose:
            import traceback
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
