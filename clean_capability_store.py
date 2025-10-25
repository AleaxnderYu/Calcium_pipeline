"""
Clean up capability store for fresh testing.

Options:
1. Backup current capabilities before cleaning
2. Clean all capabilities
3. View current capabilities before deciding
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
import sys

def view_capabilities():
    """Show current saved capabilities."""
    store_path = Path("data/capability_store")
    capabilities_dir = store_path / "capabilities"

    print("\n" + "=" * 80)
    print("CURRENT SAVED CAPABILITIES")
    print("=" * 80 + "\n")

    if not capabilities_dir.exists():
        print("No capability store found.")
        return

    cap_files = sorted(capabilities_dir.glob("*.json"))

    if not cap_files:
        print("No capabilities saved yet.")
        return

    print(f"Found {len(cap_files)} saved capabilities:\n")

    for i, meta_file in enumerate(cap_files, 1):
        try:
            metadata = json.loads(meta_file.read_text())
            cap_id = meta_file.stem

            print(f"{i}. {cap_id}")
            print(f"   Request: {metadata.get('request', 'N/A')[:70]}")
            print(f"   Created: {metadata.get('created_at', 'N/A')}")
            print(f"   Success: {metadata.get('success', False)}")
            print(f"   Reused:  {metadata.get('reuse_count', 0)} times")

            if metadata.get('last_used'):
                print(f"   Last Used: {metadata.get('last_used')}")

            print()
        except Exception as e:
            print(f"{i}. {meta_file.name} - Error reading: {e}\n")

    # Show Git history
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(store_path), "log", "--oneline", "--all"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"\nGit History ({len(lines)} commits):")
            print("-" * 80)
            for line in lines[:10]:
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... and {len(lines) - 10} more commits")
    except:
        pass


def backup_capability_store():
    """Backup the current capability store."""
    store_path = Path("data/capability_store")

    if not store_path.exists():
        print("\nNo capability store to backup.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("data/capability_store_backups")
    backup_dir.mkdir(exist_ok=True)

    backup_path = backup_dir / f"capability_store_{timestamp}"

    print(f"\nBacking up capability store to: {backup_path}")
    shutil.copytree(store_path, backup_path)
    print(f"✓ Backup created successfully!")

    # Show backup size
    total_size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
    print(f"  Size: {total_size / 1024:.1f} KB")

    return backup_path


def clean_capability_store():
    """Remove all capabilities and reset the store."""
    store_path = Path("data/capability_store")

    if not store_path.exists():
        print("\nNo capability store found. Nothing to clean.")
        return

    print("\n" + "=" * 80)
    print("CLEANING CAPABILITY STORE")
    print("=" * 80 + "\n")

    # Remove capabilities
    capabilities_dir = store_path / "capabilities"
    if capabilities_dir.exists():
        cap_files = list(capabilities_dir.glob("*"))
        print(f"Removing {len(cap_files)} capability files...")
        for f in cap_files:
            f.unlink()
        print("✓ Capabilities removed")

    # Remove ChromaDB
    db_path = store_path / "capability_db"
    if db_path.exists():
        print(f"Removing ChromaDB index...")
        shutil.rmtree(db_path)
        db_path.mkdir()
        print("✓ ChromaDB index cleared")

    # Reset Git repository
    git_dir = store_path / ".git"
    if git_dir.exists():
        print(f"Resetting Git repository...")
        shutil.rmtree(git_dir)

        # Re-initialize
        import subprocess
        subprocess.run(["git", "init"], cwd=store_path, capture_output=True)

        # Create .gitignore
        gitignore = store_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n.DS_Store\n")

        # Initial commit
        subprocess.run(["git", "add", ".gitignore"], cwd=store_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit: Reset capability store"],
            cwd=store_path,
            capture_output=True
        )
        print("✓ Git repository reset")

    print("\n✓ Capability store cleaned successfully!")
    print(f"  Location: {store_path}")


def main():
    """Interactive cleanup tool."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CAPABILITY STORE CLEANUP TOOL" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")

    # Show current state
    view_capabilities()

    print("\n" + "=" * 80)
    print("OPTIONS")
    print("=" * 80 + "\n")
    print("1. Backup current capabilities (recommended before cleaning)")
    print("2. Clean all capabilities (fresh start)")
    print("3. Backup AND clean (safest option)")
    print("4. Exit (keep everything)")

    while True:
        print("\n" + "-" * 80)
        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            backup_path = backup_capability_store()
            if backup_path:
                print(f"\n✓ Backup complete. Original store unchanged.")
                print(f"  Backup location: {backup_path}")
            break

        elif choice == "2":
            confirm = input("\n⚠️  Clean WITHOUT backup? (yes/no): ").strip().lower()
            if confirm == "yes":
                clean_capability_store()
                print("\n✓ Store cleaned. No backup was created.")
            else:
                print("\n✗ Cancelled.")
            break

        elif choice == "3":
            print("\n→ Step 1: Backing up...")
            backup_path = backup_capability_store()

            if backup_path:
                print("\n→ Step 2: Cleaning...")
                clean_capability_store()

                print("\n" + "=" * 80)
                print("✓ COMPLETE: Backed up and cleaned")
                print("=" * 80)
                print(f"\n  Backup: {backup_path}")
                print(f"  Store:  data/capability_store (now empty)")
            break

        elif choice == "4":
            print("\n✓ No changes made.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user.\n")
        sys.exit(0)
