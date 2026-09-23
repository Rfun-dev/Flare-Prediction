# Ekspor flare_baseline.ipynb menjadi PDF dan HTML.
#
# Pakai:
#   .\export_report.ps1            # PDF + HTML
#   .\export_report.ps1 -Html      # HTML saja (tidak butuh LaTeX/pandoc)
#
# Kenapa skrip ini ada: pandoc dan MiKTeX terpasang di PATH tingkat sistem,
# tapi sesi PowerShell (dan Jupyter/VS Code) yang sudah berjalan sejak sebelum
# pemasangan tidak melihatnya. Skrip ini memuat ulang PATH lebih dulu.

param([switch]$Html)

$ErrorActionPreference = "Stop"
$root   = $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$nb     = "flare_baseline.ipynb"

Set-Location $root
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path $nb)) { throw "Tidak menemukan $nb di $root" }

# Peringatkan kalau notebook belum dijalankan -- PDF-nya akan berisi kode tanpa hasil.
$cells = (Get-Content $nb -Raw -Encoding UTF8 | ConvertFrom-Json).cells
$code  = @($cells | Where-Object { $_.cell_type -eq "code" })
$withOutput = @($code | Where-Object { $_.outputs.Count -gt 0 })
Write-Host "Sel kode: $($code.Count) | punya output: $($withOutput.Count)"
if ($withOutput.Count -eq 0) {
    Write-Warning "Notebook belum pernah dijalankan. Hasil ekspor akan berisi kode TANPA output."
    Write-Warning "Buka notebook lalu 'Run All' dulu, baru jalankan skrip ini."
}

Write-Host "`n--- HTML ---"
& $python -m nbconvert --to html $nb
Write-Host "OK: flare_baseline.html"

if ($Html) { Write-Host "`nSelesai (HTML saja)."; exit 0 }

foreach ($exe in @("pandoc", "xelatex")) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        Write-Warning "$exe tidak ditemukan di PATH. Lewati PDF; pakai HTML lalu cetak ke PDF dari browser."
        exit 0
    }
}

Write-Host "`n--- PDF ---"
# nbconvert memanggil xelatex dengan -quiet sehingga error LaTeX tersembunyi.
# Karena itu tahapannya dipisah: notebook -> .tex, lalu xelatex dijalankan
# sendiri (dua kali, agar rujukan silang dan nomor halaman benar).
$work = Join-Path $env:TEMP "nbpdf"
New-Item -ItemType Directory -Force $work | Out-Null
Get-ChildItem $work -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

& $python -m nbconvert --to latex --output-dir $work $nb
Push-Location $work
1..2 | ForEach-Object { & xelatex -interaction=nonstopmode --enable-installer flare_baseline.tex | Out-Null }
Pop-Location

$pdf = Join-Path $work "flare_baseline.pdf"
if (Test-Path $pdf) {
    Copy-Item $pdf (Join-Path $root "flare_baseline.pdf") -Force
    $kb = [math]::Round((Get-Item (Join-Path $root "flare_baseline.pdf")).Length / 1KB, 1)
    Write-Host "OK: flare_baseline.pdf ($kb KB)"
    Write-Host "`nCatatan: log LaTeX memuat 'No counter none defined' beberapa kali."
    Write-Host "Itu ketidakcocokan kosmetik antara template nbconvert dan tcolorbox versi baru;"
    Write-Host "isi PDF tidak terpengaruh."
} else {
    Write-Warning "xelatex gagal. Lihat $work\flare_baseline.log"
    Write-Warning "Alternatif: buka flare_baseline.html lalu Ctrl+P -> Save as PDF."
}
