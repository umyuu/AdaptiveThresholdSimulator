cd ..\src

python -m cProfile -o ..\pref\program.prof adaptiveThreshold.py
python C:\Users\%USERNAME%\AppData\Roaming\Python\Python36\Scripts\snakeviz.exe ..\pref\program.prof
TIMEOUT 30