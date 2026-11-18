# Run in a second PowerShell terminal from the project root
Set-Location frontend
python -m http.server 5500 --bind 127.0.0.1
