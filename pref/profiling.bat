cd ..

python -m cProfile -o pref\program.prof src\simulator.py images\sakura.jpg
CALL pref\visualizer.BAT
TIMEOUT 30