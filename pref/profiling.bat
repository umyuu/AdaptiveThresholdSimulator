cd ..\src

python -m cProfile -o ..\pref\program.prof adaptiveThreshold.py
CALL ..\pref\visualizer.BAT
TIMEOUT 30