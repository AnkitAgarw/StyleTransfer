# importing only those functions 
# which are needed 

from tkinter import *

from style_transfer import style_transfer
def final_call():
    loc='./'+str(var.get())+'/'
    print(loc)
    
    style_transfer(loc)
    # creating tkinter window 
root = Tk() 
root.title("Style Transfer")


# Creating a photoimage object to use image 
photo1 = PhotoImage(file = "./1.png") 
photoimage1 = photo1.subsample(12, 7) 
photo2 = PhotoImage(file = "./2.png") 
photoimage2 = photo2.subsample(3, 4) 
photo3 = PhotoImage(file = "./3.png") 
photoimage3 = photo3.subsample(13, 7) 
photo4 = PhotoImage(file = "./4.png") 
photoimage4 = photo4.subsample(5, 3) 
photo5 = PhotoImage(file = "./5.png") 
photoimage5 = photo5.subsample(12, 6) 
photo6 = PhotoImage(file = "./6.png") 
photoimage6 = photo6.subsample(12, 6) 
photo7 = PhotoImage(file = "./7.png") 
photoimage7 = photo7.subsample(8, 4) 
photo8 = PhotoImage(file = "./8.png") 
photoimage8 = photo8.subsample(12, 7) 
photo9 = PhotoImage(file = "./9.png") 
photoimage9 = photo9.subsample(12, 7) 

# here, image option is used to 
# set image on button 
# Button(root, text = 'Click Me !', image = photoimage).pack(side = TOP) 
var=IntVar()
# Adding widgets to the root window 
def selected():
#    print(var.get())
    pass



Radiobutton(root, image=photoimage1, variable=var, value=1, command=selected,indicator = 0).grid(row=1,column=0, padx=4)
Radiobutton(root, image=photoimage2,variable=var, value=2,command=selected,indicator = 0).grid(row=1,column=1, padx=4)
Radiobutton(root, image=photoimage3, variable=var, value=3,command=selected,indicator = 0).grid(row=1,column=2,  padx=4)
Radiobutton(root, image=photoimage4, variable=var, value=4, command=selected,indicator = 0).grid(row=2,column=0,  padx=4)
Radiobutton(root, image=photoimage5, variable=var, value=5,command=selected,indicator = 0).grid(row=2,column=1,  padx=4)
Radiobutton(root, image=photoimage6, variable=var, value=6,command=selected,indicator = 0).grid(row=2,column=2, padx=4)
Radiobutton(root, image=photoimage7, variable=var, value=7,command=selected,indicator = 0).grid(row=3,column=0, padx=4)
Radiobutton(root, image=photoimage8, variable=var, value=8,command=selected,indicator = 0).grid(row=3,column=1, padx=4)
Radiobutton(root, image=photoimage9, variable=var, value=9,command=selected,indicator = 0).grid(row=3,column=2, padx=4)

Button(root, text = 'Submit',command=final_call).grid(row=4,column=1)


mainloop() 
