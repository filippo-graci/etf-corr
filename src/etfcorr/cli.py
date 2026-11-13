"""Command-line entry point (``etfcorr <step>``).

Exposes ``version``, ``check-config`` and ``lock-hypotheses`` (step 0.3), ``data
check-universe`` (step 1.1), ``data download`` (step 1.2), ``data build`` (step 1.3), ``data
holdings`` (step 2.4), ``returns`` (step 2.1) and ``eval`` (step 7.3). Each later step registers
its own command here (see project_plan.md section 6.1); nothing is added before its step is
delivered.
The commands stay thin: the logic lives in importable, typer-free functions that the tests call
directly.
"""

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from etfcorr import __version__
from etfcorr.config import (
    ConfigError,
    Paths,
    describe_project,
    find_repo_root,
    load_experiment,
    load_project,
    write_hypotheses_lock,
)
from etfcorr.data.build_panel import command_build_panel
from etfcorr.data.calendars import CalendarError, last_session
from etfcorr.data.download import download_universe, summary_lines
from etfcorr.data.holdings import HoldingsError, command_holdings
from etfcorr.data.sources import fetch_yfinance
from etfcorr.data.universe_check import command_check_universe, parse_iso_date
from etfcorr.data.validate import DataQualityError
from etfcorr.evaluation.econ import command_econ
from etfcorr.evaluation.final import EvalError, command_eval
from etfcorr.returns import command_returns

app = typer.Typer(help="Thematic ETF correlation study (research code).", no_args_is_help=True)
data_app = typer.Typer(help="Data-layer commands (Phase 1).", no_args_is_help=True)
app.add_typer(data_app, name="data")

RootOption = Annotated[
    Path | None,
    typer.Option("--root", help="Repository root (default: nearest folder with config/)."),
]
AsofOption = Annotated[
    str | None,
    typer.Option(
        "--asof",
        help="Audit date, YYYY-MM-DD (default: today). Use the last COMPLETE session (e.g. the "
        "last Friday) when running during US market hours: today's bar is unfinished.",
    ),
]
OutOption = Annotated[
    str | None,
    typer.Option(
        "--out",
        help="CSV path relative to the repository root, under data/, artifacts/ or reports/ "
        "(default: data/interim/universe_check.csv).",
    ),
]
DownloadAsofOption = Annotated[
    str | None,
    typer.Option(
        "--asof",
        help="Download date, YYYY-MM-DD (default: the last COMPLETE NYSE session on or before "
        "today, via the exchange calendar; never today's own unfinished bar).",
    ),
]
ForceOption = Annotated[
    bool,
    typer.Option("--force", help="Re-fetch every ticker even if the cache already covers asof."),
]
HoldingsYearOption = Annotated[
    int | None,
    typer.Option(
        "--year",
        help="SEC series/class dataset vintage, e.g. 2024 (default: the current year). Only "
        "affects which cached ticker-to-CIK/series mapping is built or reused.",
    ),
]
HoldingsMaxCandidatesOption = Annotated[
    int | None,
    typer.Option(
        "--max-candidates",
        help="Per-ticker cap on how many of a CIK's NPORT-P filings (newest first) to open while "
        "searching for the target series (default: 300; see fetch_sec_nport_holdings's docstring). "
        "Raise this if a ticker from a very large shared-CIK trust (e.g. iShares Trust) still "
        "fails after the default.",
    ),
]


@app.callback()
def main() -> None:
    """Thematic ETF correlation study (research code, not investment advice)."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command("check-config")
def check_config(root: RootOption = None) -> None:
    """Validate both YAML files, cross-check them and verify the pre-registration lock."""
    try:
        project = load_project(root)
    except ConfigError as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in describe_project(project):
        typer.echo(line)


@app.command("lock-hypotheses")
def lock_hypotheses(root: RootOption = None) -> None:
    """Write config/hypotheses.lock once. Refuses to overwrite an existing lock."""
    try:
        paths = Paths.from_root(root if root is not None else find_repo_root())
        exp = load_experiment(paths.experiment_yaml)
        lock = write_hypotheses_lock(exp, paths.hypotheses_lock, locked_on=date.today())
    except ConfigError as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"locked {lock['hypothesis_ids']} -> {paths.hypotheses_lock}")
    typer.echo("commit the lock together with experiment.yaml, before any result exists")


def _progress(done: int, total: int, ticker: str) -> None:
    typer.echo(f"[{done}/{total}] {ticker}", err=True)


@data_app.command("check-universe")
def data_check_universe(
    root: RootOption = None, asof: AsofOption = None, out: OutOption = None
) -> None:
    """Audit the candidate universe against yfinance (network; personal-use terms).

    Exit 0 when the report was written (failing tickers are results, not errors); 1 for an
    invalid configuration, date or output path.
    """
    try:
        project = load_project(root)
        day = date.today() if asof is None else parse_iso_date(asof)
        lines = command_check_universe(
            project, asof=day, fetch=fetch_yfinance, out=out, progress=_progress
        )
    except ConfigError as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in lines:
        typer.echo(line)


@data_app.command("download")
def data_download(
    root: RootOption = None, asof: DownloadAsofOption = None, force: ForceOption = False
) -> None:
    """Download/refresh the raw price cache (network; personal-use terms; never committed).

    Exit 0 when the panel and log were written (a failed ticker is a result, kept in the log,
    not a command failure); 1 for an invalid configuration or an unresolvable calendar date.
    """
    try:
        project = load_project(root)
        day = last_session(date.today()) if asof is None else parse_iso_date(asof)
        report = download_universe(
            project, fetch_yfinance, asof=day, force=force, progress=_progress
        )
    except (ConfigError, CalendarError) as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in summary_lines(report):
        typer.echo(line)


@data_app.command("build")
def data_build(root: RootOption = None, asof: DownloadAsofOption = None) -> None:
    """Validate the cached panel (data download) and build the clean panel + coverage table.

    Exit 0 when `data/interim/prices_clean.parquet` and `data/processed/coverage.csv` were
    written (validation warnings are printed but do not block); 1 for an invalid configuration,
    date, missing raw cache, or a validation ERROR (nothing is written in that case).
    """
    try:
        project = load_project(root)
        day = last_session(date.today()) if asof is None else parse_iso_date(asof)
        lines = command_build_panel(project, asof=day)
    except (ConfigError, CalendarError) as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DataQualityError as exc:
        typer.echo(f"DATA QUALITY ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in lines:
        typer.echo(line)


@data_app.command("holdings")
def data_holdings(
    root: RootOption = None,
    asof: DownloadAsofOption = None,
    force: ForceOption = False,
    year: HoldingsYearOption = None,
    max_candidates: HoldingsMaxCandidatesOption = None,
) -> None:
    """Fetch/refresh point-in-time ETF holdings snapshots and the holdings-overlap matrix (2.4).

    Source: SEC EDGAR N-PORT-P filings (network; only filings with filing_date <= asof are used,
    never the report's own asof, to respect the ~45 day reporting lag). Exit 0 when the snapshot
    CSVs and, when at least two tickers yielded holdings, `overlap_matrix.parquet` were written (a
    failed ticker is a result, kept in the report, not a command failure); 1 (CONFIG ERROR) for an
    invalid configuration or an unresolvable calendar date; 1 (HOLDINGS ERROR) for an unset
    SEC_USER_AGENT or a malformed overlap matrix.
    """
    try:
        project = load_project(root)
        day = last_session(date.today()) if asof is None else parse_iso_date(asof)
        lines = command_holdings(
            project,
            asof=day,
            force=force,
            year=year,
            max_candidates=max_candidates,
            progress=_progress,
        )
    except (ConfigError, CalendarError) as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except HoldingsError as exc:
        typer.echo(f"HOLDINGS ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in lines:
        typer.echo(line)


@app.command("returns")
def returns_cmd(root: RootOption = None) -> None:
    """Compute log returns at every configured horizon from the clean price panel (step 2.1).

    Exit 0 when `data/processed/returns_<horizon>.parquet` was written for every horizon in
    `config/experiment.yaml`; 1 for an invalid configuration or a missing clean panel (run
    `etfcorr data build` first).
    """
    try:
        project = load_project(root)
        lines = command_returns(project)
    except ConfigError as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in lines:
        typer.echo(line)


@app.command("eval")
def eval_cmd(
    root: RootOption = None,
    final: Annotated[
        bool,
        typer.Option("--final", help="Read the sealed TEST window. Allowed once (Phase 7)."),
    ] = False,
    confirm_unseal: Annotated[
        bool,
        typer.Option("--confirm-unseal", help="Required with --final: Filippo's go-ahead."),
    ] = False,
    recompute_dcc: Annotated[
        bool, typer.Option("--recompute-dcc", help="Ignore the cached validation DCC path.")
    ] = False,
) -> None:
    """Run the frozen ladder (rungs 0-3 with graph models) and read H3 (step 7.3).

    Without --final: the validation window (a rehearsal, about half an hour). With --final and
    --confirm-unseal: the test window, once; refused if `forecasting.final` is not frozen, the
    working tree is dirty, or an earlier final run left its marker. Exit 1 on a refusal.
    """
    try:
        project = load_project(root)
        lines = command_eval(
            project, final=final, confirm_unseal=confirm_unseal, recompute_dcc=recompute_dcc
        )
    except ConfigError as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except EvalError as exc:
        typer.echo(f"EVAL REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in lines:
        typer.echo(line)


@app.command("econ")
def econ_cmd(
    root: RootOption = None,
    final: Annotated[
        bool,
        typer.Option("--final", help="Evaluate on the TEST window. Allowed once (Phase 8)."),
    ] = False,
    confirm_unseal: Annotated[
        bool,
        typer.Option("--confirm-unseal", help="Required with --final: Filippo's go-ahead."),
    ] = False,
) -> None:
    """Portfolios (H5) and hedging (H6) on validation, or once on the test window (step 8.3).

    Exit 1 on a refusal: `economics` not frozen, a dirty tree, or an earlier final run.
    """
    try:
        project = load_project(root)
        lines = command_econ(project, final=final, confirm_unseal=confirm_unseal)
    except ConfigError as exc:
        typer.echo(f"CONFIG ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except EvalError as exc:
        typer.echo(f"EVAL REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in lines:
        typer.echo(line)


if __name__ == "__main__":
    app()
