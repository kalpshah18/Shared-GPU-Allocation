"""Compile the submission after build_final_report.py; requires PDFLaTeX/BibTeX.

The final report has eight body pages; references start on page nine.
The source archive compiles independently of the simulation repository.
"""
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "Project"
BUILD = PROJECT / "final_build"
STEM = "FAI_Project_Final_Report"


def main():
    for tool in ("pdflatex", "bibtex"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is required; install a TeX distribution or use the source in Overleaf.")
    BUILD.mkdir(exist_ok=True)
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
             "-no-shell-escape", "-output-directory=final_build", f"{STEM}.tex"]
    commands = [latex, ["bibtex", f"final_build/{STEM}"], latex, latex]
    with (BUILD / "compile-console.txt").open("w", encoding="utf-8") as log:
        for command in commands:
            print("Running", command[0], flush=True)
            result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{command[0]} failed. Inspect {BUILD / 'compile-console.txt'}")
    aux = (BUILD / f"{STEM}.aux").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\\newlabel\{lastbodypage\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not match or match.group(1) != "8":
        raise RuntimeError(f"Expected eight body pages; lastbodypage={match.group(1) if match else 'missing'}")
    log_text = (BUILD / f"{STEM}.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"Overfull \\[hv]box|undefined references|Citation .* undefined", log_text):
        raise RuntimeError("LaTeX has an overfull box or unresolved reference; inspect the log before delivery.")
    pdf = ROOT / "output/pdf/Shared_GPU_Allocation_Final_Report.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BUILD / f"{STEM}.pdf", pdf)
    archive = ROOT / "output/latex/Shared_GPU_Allocation_Final_Report_Source.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = [PROJECT / f"{STEM}.tex", PROJECT / "PRIMEarxiv.sty", PROJECT / "final_references.bib"]
    files += sorted((PROJECT / "final_assets").glob("*.tex"))
    files += sorted((PROJECT / "final_assets").glob("*.pdf"))
    files += [PROJECT / "final_assets/evidence.json"]
    if (PROJECT / "final_assets/replay_checks.json").exists():
        files += [PROJECT / "final_assets/replay_checks.json"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(PROJECT).as_posix())
        bundle.write(PROJECT / "README.md", "TEMPLATE_README.md")
        bundle.writestr("README.txt", "Compile FAI_Project_Final_Report.tex with PDFLaTeX, BibTeX and two final PDFLaTeX passes.\n"
                        "The archive includes generated assets and needs no simulator to compile.\n"
                        "Regenerate data and audit assets in the repository with python analysis/build_final_report.py.\n"
                        "Repository: https://github.com/kalpshah18/Shared-GPU-Allocation\n")
    print(f"Wrote {pdf}\nWrote {archive}")


if __name__ == "__main__":
    main()
