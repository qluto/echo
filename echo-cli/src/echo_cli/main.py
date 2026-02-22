"""Echo CLI - Offline voice transcription tool for Apple Silicon."""

import argparse
import sys

from echo_cli.database import TranscriptionDatabase


def cmd_history(args):
    db = TranscriptionDatabase()
    page = db.get_all(limit=args.limit, offset=args.offset)
    db.close()

    if not page.entries:
        print("No transcriptions yet.")
        return

    for entry in page.entries:
        dur = f"{entry.duration_seconds:.1f}s" if entry.duration_seconds else "—"
        lang = entry.language or "—"
        text_preview = entry.text[:80] + ("..." if len(entry.text) > 80 else "")
        print(f"  #{entry.id}  [{entry.created_at}]  {dur}  {lang}")
        print(f"       {text_preview}")
        print()

    shown = args.offset + len(page.entries)
    print(f"Showing {shown}/{page.total_count} transcriptions", end="")
    if page.has_more:
        print(f"  (use --offset {shown} to see more)")
    else:
        print()


def cmd_search(args):
    db = TranscriptionDatabase()
    page = db.search(args.query, limit=args.limit)
    db.close()

    if not page.entries:
        print(f'No results for "{args.query}".')
        return

    for entry in page.entries:
        dur = f"{entry.duration_seconds:.1f}s" if entry.duration_seconds else "—"
        text_preview = entry.text[:80] + ("..." if len(entry.text) > 80 else "")
        print(f"  #{entry.id}  [{entry.created_at}]  {dur}")
        print(f"       {text_preview}")
        print()

    print(f"Found {page.total_count} result(s).")


def cmd_delete(args):
    db = TranscriptionDatabase()
    if db.delete(args.id):
        print(f"Deleted transcription #{args.id}.")
    else:
        print(f"Transcription #{args.id} not found.")
    db.close()


def cmd_clear(args):
    if not args.confirm:
        print("Use --confirm to delete all transcriptions.")
        return

    db = TranscriptionDatabase()
    count = db.delete_all()
    db.close()
    print(f"Deleted {count} transcription(s).")


def cmd_export(args):
    db = TranscriptionDatabase()
    output = db.export_all(fmt=args.format)
    db.close()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Exported to {args.output}")
    else:
        print(output)


def cmd_transcribe(args):
    import json
    from pathlib import Path

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from engine import ASREngine

    engine = ASREngine(model_name=args.model)

    print(f"Loading model {args.model}...", file=sys.stderr)
    engine.load_model()

    print(f"Transcribing {audio_path.name}...", file=sys.stderr)
    # Suppress mlx-audio's stdout output during transcription
    import os
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    try:
        result = engine.transcribe(str(audio_path), args.language)
    finally:
        os.dup2(old_stdout, 1)
        os.close(devnull)
        os.close(old_stdout)

    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    text = result["text"]
    segments = result.get("segments", [])
    language = result.get("language", "")
    duration = segments[-1]["end"] if segments else None

    from echo_cli.database import TranscriptionEntry

    db = TranscriptionDatabase()
    entry_id = db.insert(
        TranscriptionEntry(
            text=text,
            duration_seconds=duration,
            language=language,
            model_name=args.model,
            segments_json=json.dumps(segments, ensure_ascii=False) if segments else None,
        )
    )
    db.close()

    print(text)
    dur_str = f"{duration:.1f}s" if duration else "—"
    print(f"\nSaved as #{entry_id} ({dur_str}, {language})", file=sys.stderr)


def cmd_listen(args):
    import logging

    from silero_vad import load_silero_vad

    from engine import ASREngine
    from echo_cli.database import TranscriptionDatabase
    from echo_cli.listener import ContinuousListener

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"Loading ASR model ({args.model})...", file=sys.stderr)
    engine = ASREngine(model_name=args.model)
    result = engine.load_model()
    if not result.get("success"):
        print(f"Error loading ASR model: {result.get('error')}", file=sys.stderr)
        sys.exit(1)

    print("Loading VAD model...", file=sys.stderr)
    vad = load_silero_vad()

    db = TranscriptionDatabase()

    listener = ContinuousListener(
        engine=engine,
        vad_model=vad,
        database=db,
        language=args.language,
        silence_sec=args.silence_sec,
        max_segment_sec=args.max_segment_sec,
    )

    try:
        listener.run()
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="echo",
        description="Offline voice transcription tool for Apple Silicon",
    )
    subparsers = parser.add_subparsers(dest="command")

    # transcribe
    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    p_transcribe.add_argument("audio_path", help="Path to audio file (WAV, etc.)")
    p_transcribe.add_argument("--language", default=None, help="Language code (ja, en, etc.)")
    p_transcribe.add_argument(
        "--model",
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
        help="ASR model name",
    )

    # listen
    p_listen = subparsers.add_parser("listen", help="Continuous listening mode")
    p_listen.add_argument("--language", default=None, help="Language code (ja, en, etc.)")
    p_listen.add_argument(
        "--model",
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
        help="ASR model name",
    )
    p_listen.add_argument(
        "--silence-sec",
        type=float,
        default=1.5,
        help="Silence duration to end a segment (default: 1.5)",
    )
    p_listen.add_argument(
        "--max-segment-sec",
        type=int,
        default=60,
        help="Max segment duration before forced split (default: 60)",
    )

    # history
    p_history = subparsers.add_parser("history", help="Show transcription history")
    p_history.add_argument("--limit", type=int, default=20, help="Number of entries")
    p_history.add_argument("--offset", type=int, default=0, help="Offset for pagination")

    # search
    p_search = subparsers.add_parser("search", help="Full-text search transcriptions")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=20, help="Max results")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a transcription")
    p_delete.add_argument("id", type=int, help="Transcription ID")

    # clear
    p_clear = subparsers.add_parser("clear", help="Delete all transcriptions")
    p_clear.add_argument("--confirm", action="store_true", help="Confirm deletion")

    # export
    p_export = subparsers.add_parser("export", help="Export transcriptions")
    p_export.add_argument(
        "--format", choices=["json", "txt"], default="json", help="Output format"
    )
    p_export.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    commands = {
        "transcribe": cmd_transcribe,
        "listen": cmd_listen,
        "history": cmd_history,
        "search": cmd_search,
        "delete": cmd_delete,
        "clear": cmd_clear,
        "export": cmd_export,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
