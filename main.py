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
from utils.logging_config import setup_logging
from graph.workflow import run_workflow


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

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(
        log_level=config.LOG_LEVEL,
        log_file=config.LOG_FILE,
        verbose=args.verbose
    )

    # Print header
    print("=" * 50)
    print("CALCIUM IMAGING ANALYSIS")
    print("=" * 50)
    print(f"Request: {args.request}")
    print(f"Images: {args.images}")
    print()

    try:
        # Rebuild RAG if requested
        if args.rebuild_rag:
            print("Rebuilding RAG vector database...")
            from layers.rag_system import RAGSystem
            rag = RAGSystem()
            rag.rebuild()
            print("✓ RAG database rebuilt\n")

        # Run workflow
        print("[Step 1/5] Preprocessing...", end=" ", flush=True)
        print("[Step 2/5] Retrieving methods...", end=" ", flush=True)
        print("[Step 3/5] Generating code...", end=" ", flush=True)
        print("[Step 4/5] Executing analysis...", end=" ", flush=True)
        print("[Step 5/5] Formatting output...", end=" ", flush=True)

        # Actually run the workflow (progress is logged, not printed)
        result = run_workflow(
            user_request=args.request,
            images_path=args.images
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
