@echo off
echo Building pgn_processor with MSVC...
cl /std:c++17 /O2 /EHsc pgn_processor.cpp stats.cpp ../utils/utils.cpp
if %ERRORLEVEL% == 0 (
    echo Build successful! Run pgn_processor.exe to start.
) else (
    echo Build failed. Make sure you are running this from the Developer Command Prompt.
)
