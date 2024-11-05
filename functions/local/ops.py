import os
import subprocess as sp

commands = {
    "Pycharm":"pycharm.sh",
    "Android Studio":"androidstudio.sh",
    "Webstorm":"webstorm.sh",
    "Arduino":"arduino-ide.sh",
}

def execute(command):
    os.system(command)

# def open_pycharm():
#     os.system("pycharm.sh")
#
# def open_androidstudio():
#     os.system("androidstudio.sh")
#
# def open_webstorm():
#     os.system("webstorm.sh")
#
# def open_arduino():
#     os.system("arduino-ide.sh")