
from tkinter import *
#from PIL import ImageTk



color2 = 'dark slate gray'
color1 = 'black'
blanco = "white"
margin_left = 50
margin_up = 50
desp_y = 20

def guindou():
    def click():
        prueba1 = bool(var1.get())
        prueba2 = bool(var2.get())
        prueba3 = bool(var3.get())
        prueba4 = bool(var.get())
        print("---------")
        if not prueba4:
            print(var.get(), " ++++++")
            name = video.get()
            print(name)
        print(var1.get(), "-----1")
        print(var2.get(), "-----2")
        print(var3.get(), "-----3")
        window.quit()
    window = Tk()
    window.title("Anomaly Detection System")
    window_width = 500
    window_height = 300
    window.geometry(f"{window_width}x{window_height}")
    window.config(background=color1)
    Label(window, text="Models:", bg=color1, fg=blanco).place(
        x=margin_left, y=margin_up)
    var1 = IntVar()
    Checkbutton(window, text="ResNet50 \t", variable=var1, width=0).place(
        x=margin_left, y=margin_up+desp_y)
    var2 = IntVar()
    Checkbutton(window, text="InceptionV3\t", variable=var2, width=0).place(
        x=margin_left, y=margin_up+desp_y*2)
    var3 = IntVar()
    Checkbutton(window, text="FrankensNet\t", variable=var3, width=0).place(
        x=margin_left, y=margin_up+desp_y*3)

    Label(window, text="Video:", bg=color1, fg=blanco).place(
        x=margin_left+200, y=margin_up)
    var = IntVar()
    video = Entry(window,width = 19, fg=color1)
    video.place(x=margin_left + 200, y=margin_up+desp_y)
    R1 = Radiobutton(window, text="Choose one\t", variable=var,
                    value=0)
    R1.place(x=margin_left + 200, y=margin_up+desp_y*2)
    R2 = Radiobutton(window, text="All videos(default)\t",
                    variable=var, value=1)
    R2.place(x=margin_left + 200, y=margin_up+desp_y*3)

    Button(window, text='Quit', command=window.quit).place(x=margin_left, y=margin_up + 100)
    Button(window, text='Enviar', command=click).place(x=margin_left+200, y=margin_up + 100)


    window.mainloop()
guindou()