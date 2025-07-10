"""
main.py

Entry point for the Universal Quantum Circuit Simulator CLI.
Initializes the QPU and launches the interactive CLI.
"""

from cli import interactive_cli


def main():
    """
    Launch the simulator CLI.

    Ensures:
         The interactive CLI is started.
    """
    print("=== Starting Universal Quantum Circuit Simulator CLI ===")
    interactive_cli()


if __name__ == '__main__':
    main()
