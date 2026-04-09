import bungee
print(dir(bungee))
try:
    print(dir(bungee.Bungee))
    print(bungee.Bungee.__doc__)
except AttributeError:
    pass
try:
    print(help(bungee.pitch_shift))
except AttributeError:
    pass
