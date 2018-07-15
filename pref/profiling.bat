cd ..\src

python -m cProfile -o ..\pref\program.prof simulator.py
CALL ..\pref\visualizer.BAT
TIMEOUT 30