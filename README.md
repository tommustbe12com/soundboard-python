# soundboard-python
Soundboard made in python. Executable can be downloaded on https://tommustbe12.com/soundboard, or you can compile from source.

Config file (soundboard_config.json) is created or read on run, and then is used for reference every time you load the soundboard. if you are using as an exe, don't delete this file it will delete all your mp3 links (ur mp3 files will still exist, but soundboard won't actually have them anymore)

have fun, i was sick of spyware soundboards and finally got around to making one

NOTES:
Compile to exe with pyinstaller:
pyinstaller --onefile --windowed --icon=icon.ico soundboard.py