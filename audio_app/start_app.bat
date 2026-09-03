@echo off
chcp 65001 >nul
echo ========================================
echo 🎵 音频内容识别系统
echo ========================================
echo.

cd /d %~dp0

REM 尝试常见的Anaconda安装路径
set CONDA_PATH=
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set CONDA_PATH=%USERPROFILE%\anaconda3
) else if exist "%USERPROFILE%\Anaconda3\Scripts\conda.exe" (
    set CONDA_PATH=%USERPROFILE%\Anaconda3
) else if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" (
    set CONDA_PATH=C:\ProgramData\anaconda3
) else if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
    set CONDA_PATH=C:\ProgramData\Anaconda3
) else if exist "C:\anaconda3\Scripts\conda.exe" (
    set CONDA_PATH=C:\anaconda3
) else if exist "C:\Anaconda3\Scripts\conda.exe" (
    set CONDA_PATH=C:\Anaconda3
)

if defined CONDA_PATH (
    echo ✅ 找到Anaconda: %CONDA_PATH%
    echo 🔄 激活conda环境 d25_t6...
    call "%CONDA_PATH%\Scripts\activate.bat" d25_t6
    if errorlevel 1 (
        echo ⚠️  无法激活conda环境，使用base环境...
        call "%CONDA_PATH%\Scripts\activate.bat"
    )
) else (
    echo ⚠️  未找到Anaconda，使用系统Python...
    echo.
    echo 💡 提示：如果需要使用conda环境，请：
    echo    1. 打开 "Anaconda Prompt"
    echo    2. 运行: conda activate d25_t6
    echo    3. 运行: python "%~dp0run_app.py"
    echo.
)

echo.
echo 📋 检查依赖...
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo ⚠️  检测到缺少gradio，正在安装...
    pip install gradio "huggingface-hub>=0.33.5"
    echo.
)

echo 🚀 启动应用...
echo.
echo 💡 访问地址: http://localhost:7860
echo 💡 按 Ctrl+C 停止服务
echo.
echo ========================================
echo.

REM 解决OpenMP库冲突问题
set KMP_DUPLICATE_LIB_OK=TRUE

python run_app.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ❌ 启动失败！
    echo ========================================
    echo.
    echo 💡 建议使用 Anaconda Prompt：
    echo    1. 打开 "Anaconda Prompt"
    echo    2. 运行: conda activate d25_t6
    echo    3. 运行: D:
    echo    4. 运行: cd %~dp0
    echo    5. 运行: python run_app.py
    echo.
)

pause
