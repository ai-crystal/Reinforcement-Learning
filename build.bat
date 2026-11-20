@echo off
chcp 65001 >nul
setlocal
title ShiCuo - Build
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

rem ---- python ----
set "PY=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -c "import pymupdf, PIL" >nul 2>&1
if errorlevel 1 (
    echo [info] installing pymupdf pillow ...
    "%PY%" -m pip install --quiet pymupdf pillow
)

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

if /i "%TARGET%"=="all" goto :all
if /i "%TARGET%"=="figures" goto :figures
if /i "%TARGET%"=="checks" goto :checks_only
if /i "%TARGET%"=="concepts" goto :one_arg
if /i "%TARGET%"=="hands-on" goto :one_arg
if /i "%TARGET%"=="algorithms" goto :one_arg
if /i "%TARGET%"=="llm-rl" goto :one_arg

echo Unknown target: %TARGET%
echo Use: concepts / hands-on / algorithms / llm-rl / figures / checks / all
pause
exit /b 1

:all
call :one concepts
if errorlevel 1 goto :fail
call :one hands-on
if errorlevel 1 goto :fail
call :one algorithms
if errorlevel 1 goto :fail
call :one llm-rl
if errorlevel 1 goto :fail
goto :done

:one_arg
call :one %TARGET%
if errorlevel 1 goto :fail
goto :done

:one
echo.
echo ==================================================
echo   Building %~1
echo ==================================================
"%PY%" books\build\checks.py --book books\%~1
if errorlevel 1 exit /b 1
"%PY%" books\build\build.py --book books\%~1
if errorlevel 1 exit /b 1
exit /b 0

:checks_only
"%PY%" books\build\checks.py --all
if errorlevel 1 goto :fail
echo All books passed.
pause
exit /b 0

:figures
echo [figures] rendering ...
for /d %%V in (books\code\*) do (
    for /d %%D in ("%%V\ch*") do (
        if exist "%%D\make_figures.py" (
            echo   - %%V\%%~nxD
            "%PY%" "%%D\make_figures.py" >nul 2>&1
            if errorlevel 1 echo     [!] failed: %%D
        )
    )
)
echo [figures] done
pause
exit /b 0

:done
echo.
echo ==================================================
echo   Done. PDFs are in output\
echo ==================================================
pause
exit /b 0

:fail
echo.
echo [failed] see the messages above.
pause
exit /b 1
