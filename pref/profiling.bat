cd ..\src

python -m cProfile -o ..\pref\program.prof adaptiveThreshold.py
snakeviz ..\pref\program.prof
TIMEOUT 30