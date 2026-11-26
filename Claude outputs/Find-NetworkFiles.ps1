<#
.SYNOPSIS
  Finds files whose names contain "network" or "servers" on every fixed drive,
  and checks the Recycle Bin too. Optionally also searches inside text files.

.EXAMPLE
  .\Find-NetworkFiles.ps1
  .\Find-NetworkFiles.ps1 -Content          # also look inside text files (slower)
  .\Find-NetworkFiles.ps1 -Terms network,servers,ip,map
  .\Find-NetworkFiles.ps1 -OutDir D:\Reports              # results somewhere else (default: this script's folder)
#>
param(
    [string[]]$Terms   = @('network', 'servers'),
    [switch]  $Content,                                   # also search inside text files
    [string]  $OutDir  = $PSScriptRoot                    # default: the folder this script is in
)

# Folders that are huge and won't hold your own files
$skip = '\\Windows\\|\\Program Files|\\ProgramData\\|\\AppData\\Local\\(Temp|Microsoft|Packages)|\\node_modules\\|\\\.git\\|\\\.venv\\|\\site-packages\\'
$namePattern = ($Terms | ForEach-Object { [regex]::Escape($_) }) -join '|'
$textExt     = '.txt', '.md', '.csv', '.log', '.ini', '.conf', '.cfg', '.yaml', '.yml', '.json', '.xml', '.ps1', '.sh'
$ipRegex     = '\b(?:\d{1,3}\.){3}\d{1,3}\b'

if (-not $OutDir) { $OutDir = (Get-Location).Path }
New-Item -ItemType Directory -Path $OutDir -Force -ErrorAction Stop | Out-Null
$OutCsv = Join-Path $OutDir 'network_file_search.csv'
$OutTxt = Join-Path $OutDir 'network_file_search.txt'

$results = [System.Collections.Generic.List[object]]::new()
$drives  = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -gt 0 -and $_.Root -match '^[A-Z]:\\$' }

foreach ($d in $drives) {
    Write-Host "Searching $($d.Root) ..." -ForegroundColor Cyan
    Get-ChildItem -Path $d.Root -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch $skip } |
        ForEach-Object {
            $why = $null
            if ($_.Name -match $namePattern) {
                $why = 'name'
            }
            elseif ($Content -and $textExt -contains $_.Extension.ToLower() -and $_.Length -lt 5MB) {
                $text = Get-Content -LiteralPath $_.FullName -Raw -ErrorAction SilentlyContinue
                if ($text -match $namePattern) {
                    $why = 'content'
                }
                elseif ($text -and ([regex]::Matches($text, $ipRegex)).Count -ge 3) {
                    $why = 'contains IP addresses'
                }
            }
            if ($why) {
                $results.Add([pscustomobject]@{
                    Match    = $why
                    Name     = $_.Name
                    Folder   = $_.DirectoryName
                    Modified = $_.LastWriteTime
                    SizeKB   = [math]::Round($_.Length / 1KB, 1)
                })
            }
        }
}

# Recycle Bin: deleted files keep their original names here
Write-Host "Checking the Recycle Bin ..." -ForegroundColor Cyan
$bin = (New-Object -ComObject Shell.Application).NameSpace(0xA)
foreach ($item in $bin.Items()) {
    if ($item.Name -match $namePattern) {
        $results.Add([pscustomobject]@{
            Match    = 'RECYCLE BIN'
            Name     = $item.Name
            Folder   = $bin.GetDetailsOf($item, 1)   # original location
            Modified = $bin.GetDetailsOf($item, 2)   # date deleted
            SizeKB   = $null
        })
    }
}

if ($results.Count -eq 0) {
    Write-Host "Nothing found." -ForegroundColor Yellow
    "Search for: $($Terms -join ', ')   Run: $(Get-Date -Format 'yyyy-MM-dd HH:mm')   Nothing found." |
        Out-File -FilePath $OutTxt -Encoding UTF8
    return
}

$results = $results | Sort-Object Modified -Descending
$results | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8 -ErrorAction Stop

# Plain text: a header, then one block per file with its full path
$lines = @(
    "Search for: $($Terms -join ', ')   (content search: $([bool]$Content))"
    "Run: $(Get-Date -Format 'yyyy-MM-dd HH:mm')   Matches: $($results.Count)"
    ''
)
foreach ($r in $results) {
    $lines += Join-Path $r.Folder $r.Name
    $lines += "    match: $($r.Match)   modified: $($r.Modified)   size: $($r.SizeKB) KB"
}
$lines | Out-File -FilePath $OutTxt -Encoding UTF8 -ErrorAction Stop

Write-Host "$($results.Count) matches. Saved to $OutTxt and $OutCsv" -ForegroundColor Green
$results | Out-GridView -Title "Files matching: $($Terms -join ', ')"
