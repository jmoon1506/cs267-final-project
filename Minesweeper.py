import eel, random

eel.init('web')                     # Give folder containing web files

@eel.expose
def solve_next():
    return random.random()

eel.start('index.html', options={'mode': 'chrome', 'port': 8001})    # Start
