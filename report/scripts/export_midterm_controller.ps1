$ErrorActionPreference = 'Stop'
$deckStem = 'DASH01_controller_midterm_Tim_template'
$workspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$deckFolder = Join-Path $workspaceRoot 'report/presentations/midterm-2026-10-29'
$previewFolder = Join-Path $deckFolder 'preview'
New-Item -ItemType Directory -Force -Path $previewFolder | Out-Null
$deckApp = New-Object -ComObject PowerPoint.Application
$controllerDeck = $null
try {
    $controllerDeck = $deckApp.Presentations.Open((Join-Path $deckFolder ($deckStem + '.pptx')), -1, 0, 0)
    $overflowRecords = @()
    $inheritedBounds = @()
    for ($slideIndex = 1; $slideIndex -le $controllerDeck.Slides.Count; $slideIndex++) {
        $controllerDeck.Slides.Item($slideIndex).Export((Join-Path $previewFolder ('slide-{0:D2}.png' -f $slideIndex)), 'PNG', 1600, 900)
        foreach ($slideShape in $controllerDeck.Slides.Item($slideIndex).Shapes) {
            if ($slideShape.HasTextFrame -and $slideShape.TextFrame.HasText) {
                $excessHeight = $slideShape.TextFrame.TextRange.BoundHeight - $slideShape.Height
                $excessWidth = $slideShape.TextFrame.TextRange.BoundWidth - $slideShape.Width
                if (($excessHeight -gt 3) -or ($excessWidth -gt 3)) {
                    $record = [PSCustomObject]@{ slide=$slideIndex; text=$slideShape.TextFrame.TextRange.Text; excessHeight=$excessHeight; excessWidth=$excessWidth }
                    if (($slideIndex -in @(1,18)) -and ($record.text -in @('DASH-01','Questions?')) -and ($excessHeight -le 6.1) -and ($excessWidth -le 0)) {
                        $inheritedBounds += $record
                    } else { $overflowRecords += $record }
                }
            }
        }
    }
    ConvertTo-Json -InputObject @($overflowRecords) -Depth 3 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $previewFolder 'text-bounds.json')
    ConvertTo-Json -InputObject @($inheritedBounds) -Depth 3 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $previewFolder 'inherited-template-bounds.json')
    $controllerDeck.SaveAs((Join-Path $deckFolder ($deckStem + '.pdf')), 32)
    Write-Output "Exported $($controllerDeck.Slides.Count) slide previews and PDF"
    Write-Output "New content overflow records: $($overflowRecords.Count)"
} finally {
    if ($null -ne $controllerDeck) {
        $controllerDeck.Close()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($controllerDeck) | Out-Null
    }
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($deckApp) | Out-Null
}
