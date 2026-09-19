import sys
import time

def ketik(text, delay=0.067):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def muncul(text, delay=0.067):
    ketik(text, delay)
    time.sleep(0.5)