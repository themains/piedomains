"""Command-line interface for piedomains."""

import argparse
import json
import sys


def _read_domains(args: argparse.Namespace) -> list[str]:
    """Collect domains from positional arguments or an input file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        list[str]: Domains to classify, in the order given.

    Raises:
        SystemExit: If no domains were supplied.
    """
    domains = list(args.domains)
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            domains.extend(
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("#")
            )
    if not domains:
        raise SystemExit("No domains supplied. Pass domains directly or use --file.")
    return domains


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="classify_domains",
        description="Classify websites into content categories.",
    )
    parser.add_argument("domains", nargs="*", help="Domain names or URLs to classify")
    parser.add_argument("--file", help="File with one domain per line")
    parser.add_argument(
        "--method",
        default="combined",
        choices=["combined", "text", "images"],
        help="Classification method (default: combined)",
    )
    parser.add_argument(
        "--archive-date",
        help="Analyze an archive.org snapshot near this date (YYYYMMDD)",
    )
    parser.add_argument("--cache-dir", help="Override the cache directory")
    parser.add_argument(
        "--output",
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--report",
        help="Write the run report (counts by reason, missing domains) to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    domains = _read_domains(args)

    from .api import classify_domains

    run = classify_domains(
        domains,
        method=args.method,
        archive_date=args.archive_date,
        cache_dir=args.cache_dir,
    )
    results = run["results"]
    report = run["report"]

    if args.output == "json":
        json.dump(run, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for result in results:
            confidence = result.get("confidence")
            formatted = f"{confidence:.3f}" if isinstance(confidence, float) else "n/a"
            sys.stdout.write(
                f"{result.get('domain')}\t{result.get('status')}\t"
                f"{result.get('category')}\t{formatted}\t"
                f"{result.get('error_code') or ''}\n"
            )

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)

    # Summary goes to stderr so it never contaminates piped results.
    sys.stderr.write(
        f"\n{report['classified']}/{report['total']} classified, "
        f"{report['failed']} failed (run {report['run_id']})\n"
    )
    for reason, count in report["by_reason"].items():
        sys.stderr.write(f"  {reason}: {count}\n")
    if report["missing"]:
        preview = ", ".join(report["missing"][:10])
        more = (
            "" if len(report["missing"]) <= 10 else f" (+{len(report['missing']) - 10})"
        )
        sys.stderr.write(f"  no result for: {preview}{more}\n")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
