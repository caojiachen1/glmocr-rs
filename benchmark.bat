@echo off
setlocal enabledelayedexpansion

set RUNS=5
set BACKEND=
set IMAGE=test.png
set EXE=target\release\glm_ocr_onnx_rust.exe
set TMPF=%TEMP%\bench_out.txt

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--backend" (
    set BACKEND=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--image" (
    set IMAGE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--runs" (
    set RUNS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-h" goto usage
if "%~1"=="--help" goto usage
echo [BENCH] Unknown argument: %~1
exit /b 1

:usage
echo Usage: benchmark.bat [--backend onnx^|native^|gguf] [--image path] [--runs N]
echo   Default: test all backends with 5 runs each
exit /b 0

:args_done
if not exist "%EXE%" (
    echo [BENCH] Error: %EXE% not found. Run cargo build --release first.
    exit /b 1
)

if "%BACKEND%"=="" (
    echo [BENCH] Testing all backends, %RUNS% runs each
    echo.
    call :bench_backend onnx
    echo.
    call :bench_backend native
    echo.
    call :bench_backend gguf
    echo.
    echo [BENCH] ========================================
    echo [BENCH]  Comparison
    echo [BENCH] ========================================
    if defined BEST_NAME (
        echo [BENCH]  Winner: !BEST_NAME! ^(!BEST_TPS! tok/s, !BEST_ELAPSED!^)
    )
) else (
    call :bench_backend %BACKEND%
)

del "%TMPF%" 2>nul
endlocal
exit /b 0

:bench_backend
set "BNAME=%~1"
set "BMIN=999999"
set "BMAX=0"
set "BSUM=0"
set "BTOKENS=0"

echo [BENCH] Backend: %BNAME%  Runs: %RUNS%

echo [BENCH]  Warmup...
"%EXE%" --backend %BNAME% --image %IMAGE% --timing >nul 2>nul

for /l %%i in (1,1,%RUNS%) do (
    set "elapsed_val="
    set "tokens_val="

    "%EXE%" --backend %BNAME% --image %IMAGE% --timing >"%TMPF%" 2>nul
    set /p RAWLINE=<"%TMPF%"

    if defined RAWLINE (
        for /f "tokens=1,2,3" %%x in ("!RAWLINE!") do (
            set "P1=%%x"
            set "P2=%%y"
            set "P3=%%z"
        )

        if "!P1:~0,8!"=="ELAPSED=" (
            set "elapsed_val=!P1:~8!"
        )
        if "!P2:~0,8!"=="ELAPSED=" (
            set "elapsed_val=!P2:~8!"
        )
        if "!P3:~0,8!"=="ELAPSED=" (
            set "elapsed_val=!P3:~8!"
        )

        if "!P1:~0,7!"=="TOKENS=" set "tokens_val=!P1:~7!"
        if "!P2:~0,7!"=="TOKENS=" set "tokens_val=!P2:~7!"
        if "!P3:~0,7!"=="TOKENS=" set "tokens_val=!P3:~7!"

        if defined elapsed_val (
            set "elapsed_val=!elapsed_val:s=!"
        )
    )

    if not defined elapsed_val (
        echo [BENCH]  Run %%i: no timing data
    ) else (
        if not defined tokens_val set tokens_val=0
        set "BTOKENS=!tokens_val!"

        for /f "tokens=1,2 delims=." %%x in ("!elapsed_val!") do (
            set "int_part=%%x"
            set "frac_part=%%y"
        )
        if not defined frac_part set frac_part=0
        set "frac_part=!frac_part!000"
        set "frac_part=!frac_part:~0,3!"
        set /a "elapsed_ms=!int_part!*1000+1!frac_part!-1000"

        if !elapsed_ms! gtr 0 (
            set /a "tps_x100=!tokens_val!*100000/!elapsed_ms!"
            set /a "tps_int=!tps_x100!/100"
            set /a "tps_frac=!tps_x100!%%100"
            if !tps_frac! lss 10 set tps_frac=0!tps_frac!
        ) else (
            set tps_int=0
            set tps_frac=00
        )

        echo [BENCH]  Run %%i: !elapsed_val!s  ^(!tokens_val! tokens, !tps_int!.!tps_frac! tok/s^)

        if !elapsed_ms! lss !BMIN! set BMIN=!elapsed_ms!
        if !elapsed_ms! gtr !BMAX! set BMAX=!elapsed_ms!
        set /a BSUM+=!elapsed_ms!
    )
)

set /a BAVG=%BSUM%/%RUNS%
call :fmt_ms %BMIN% FMIN
call :fmt_ms %BMAX% FMAX
call :fmt_ms %BAVG% FAVG

set /a "TPS_X100=!BTOKENS!*100000/!BAVG!"
set /a "TPS_I=!TPS_X100!/100"
set /a "TPS_F=!TPS_X100!%%100"
if !TPS_F! lss 10 set TPS_F=0!TPS_F!

echo [BENCH]  Summary: min=%FMIN%  max=%FMAX%  avg=%FAVG%  ^(!TPS_I!.!TPS_F! tok/s^)

if not defined BEST_NAME (
    set "BEST_NAME=%BNAME%"
    set "BEST_TPS=!TPS_I!.!TPS_F!"
    set "BEST_ELAPSED=%FAVG%"
    set "BEST_AVG_MS=!BAVG!"
) else (
    if !BAVG! lss !BEST_AVG_MS! (
        set "BEST_NAME=%BNAME%"
        set "BEST_TPS=!TPS_I!.!TPS_F!"
        set "BEST_ELAPSED=%FAVG%"
        set "BEST_AVG_MS=!BAVG!"
    )
)

exit /b 0

:fmt_ms
set /a "s=%1/1000"
set /a "ms=%1%%1000"
if %ms% lss 100 (
    if %ms% lss 10 (
        set "ms=00%ms%"
    ) else (
        set "ms=0%ms%"
    )
)
set "%2=%s%.%ms%s"
exit /b 0
