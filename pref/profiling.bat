cd ..\src

python -m cProfile -o ..\pref\program.prof simulator.py ..\images\sakura.jpg
CALL ..\pref\visualizer.BAT
TIMEOUT 30