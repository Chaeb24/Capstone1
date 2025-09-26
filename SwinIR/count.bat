@echo off
REM Python 실행 경로 (필요시 수정)
set PYTHON=python

%PYTHON% count.py

echo ==============================
echo PSNR/SSIM comparison completed!
echo Result file: result.csv
echo ==============================

pause