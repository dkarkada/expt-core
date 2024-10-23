import matplotlib.pyplot as plt


def rcsetup():    
    plt.rc("figure", dpi=120, facecolor=(1, 1, 1))
    plt.rc("font", family='stixgeneral', size=12)
    plt.rc("axes", facecolor=(1, .99, .95), titlesize=12)
    plt.rc("mathtext", fontset='cm')


class Logger:

    def __init__(self, filename=None):
        self.filename = filename        
        if filename:
            with open(filename, 'w') as _:
                pass
    
    def print(self, msg='', end='\n'):
        print(msg, end=end)
        if self.filename:
            with open(self.filename, 'a') as file:
                file.write(f'{msg}{end}')


