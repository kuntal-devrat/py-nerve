"""Interactive desktop-automation agent.

Type a task in plain English; the Py-Nerve agent sees your screen via OCR,
plans the steps with Ollama (or any OpenAI-compatible endpoint), and executes
them deterministically with the native input engine. No mouse/keyboard is
touched unless the model calls a tool — and ``--dry-run`` prevents that too.

Examples:
    python scripts/desktop_agent_cli.py                          # interactive
    python scripts/desktop_agent_cli.py --dry-run                # plan only, safe
    python scripts/desktop_agent_cli.py --model qwen2.5:7b       # pick a model
    python scripts/desktop_agent_cli.py --endpoint http://localhost:1234/v1  # LM Studio
    python scripts/desktop_agent_cli.py "Open Chrome, go to youtube.com and play lofi beats"
"""

from __future__ import annotations

import argparse
import sys

from pynerve import Agent, AgentConfig


def _print_result(result) -> None:
    print("\n--- transcript ---")
    for entry in result.transcript:
        print(f"  step {entry['step']}: {entry['tool']}({entry['args']})")
    print("--- result ---")
    print(result.final_answer)
    print()


def main(argv: list[str] | None = None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # LLM output can contain characters the Windows console (cp1252) can't
    # encode; replace them instead of crashing on print().
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
        sys.stderr.reconfigure(errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(
        description="Interactive desktop automation agent (Ollama or any OpenAI-compatible endpoint)",
    )

    parser.add_argument("--model", default="qwen2.5-coder:1.5b", help="Model tag (see `ollama list`)")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL (Ollama default; LM Studio is http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the endpoint (cloud providers). Falls back to OPENROUTER_API_KEY / PYNERVE_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY",
    )
    parser.add_argument("--max-steps", type=int, default=15, help="Max tool calls before giving up")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning budget for reasoning models (e.g. 'low' for gpt-oss on Groq — saves tokens)",
    )
    parser.add_argument(
        "--max-obs-elements",
        type=int,
        default=150,
        help="Cap on on-screen elements sent to the model per observe (lower = fewer tokens)",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between LLM requests (e.g. 15.0 to stay within Groq free-tier rate limits)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only: print actions, never touch the mouse")
    parser.add_argument("task", nargs="*", help="Optional task to run once and exit (omit for interactive mode)")
    args = parser.parse_args(argv)

    def run(task: str) -> None:
        config = AgentConfig(
            model=args.model,
            base_url=args.endpoint,
            api_key=args.api_key,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
            reasoning_effort=args.reasoning_effort,
            max_observe_elements=args.max_obs_elements,
            step_delay=args.step_delay,
        )
        result = Agent(config=config).run(task)
        _print_result(result)


    if args.task:
        try:
            run(" ".join(args.task))
        except KeyboardInterrupt:
            print("\n[Agent aborted by user via Ctrl+C]")
            return 130
        return 0

    mode = "  [DRY-RUN: no mouse or keyboard will be touched]" if args.dry_run else ""
    print(f"Py-Nerve desktop agent — model={args.model} endpoint={args.endpoint}{mode}")
    print("Type a task in plain English and press Enter. Empty line or Ctrl+C to quit.")
    print('Example: "Open Chrome, go to youtube.com and play my favorite lofi mix"')
    while True:
        try:
            task = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not task:
            continue
        try:
            run(task)
        except KeyboardInterrupt:
            print("\n[Task aborted by user via Ctrl+C]")
            continue
        except Exception as e:  # keep the session alive on errors
            print(f"ERROR: {type(e).__name__}: {e}\n")



if __name__ == "__main__":
    sys.exit(main())
