$ErrorActionPreference = 'Stop'

Write-Host 'Gemma API secure setup' -ForegroundColor Cyan
Write-Host 'Paste a NEW rotated key only. Input will be hidden.' -ForegroundColor Yellow

$secureKey = Read-Host 'Gemma API key' -AsSecureString
$plainKeyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($plainKeyPtr).Trim()
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($plainKeyPtr)
}

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw 'No API key was entered.'
}

$model = Read-Host 'Gemini/Gemma model (default: gemini-3.1-flash-lite)'
if ([string]::IsNullOrWhiteSpace($model)) { $model = 'gemini-3.1-flash-lite' }

$envPath = Join-Path $PSScriptRoot '.env'
$envContent = "GEMMA_API_KEY=$apiKey`r`nGEMMA_MODEL=$model`r`n"
[System.IO.File]::WriteAllText($envPath, $envContent, [System.Text.UTF8Encoding]::new($false))

Write-Host 'Configured locally in .env. .env is ignored by Git.' -ForegroundColor Green
Write-Host 'Restart the application so python-dotenv loads the new key.' -ForegroundColor Cyan
