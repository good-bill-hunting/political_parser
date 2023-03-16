from IPython.display import Audio, display

def play_sound(self, etype, value, tb, tb_offset=None):
    self.showtraceback((etype, value, tb), tb_offset=tb_offset)
    display(Audio(url='problem.wav', autoplay=True))
    
#Paste this code into the first line of the notebook:
#get_ipython().set_custom_exc((Exception,), play_sound)
    
def final_ding():
    display(Audio("ding.mp3", autoplay=True))