@echo off

set i=0

:: Loop from 1 to 129
for /L %%i in (1, 1, 129) do (
    :: Run BHVDimChange.py in order to update BHV Dimension
    python BHVDimChange.py %%i

    :: Run UCR_SVM.py with the current benchmark
    python UCR_SVM.py %%i
)

:: Pause to keep the command prompt window open (optional)
pause