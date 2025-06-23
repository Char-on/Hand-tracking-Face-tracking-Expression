# Hand-tracking-Face-tracking-Expression
Hello, I am an amateur python coder, and i am currently 16 and i would also love your feedbacks towards my code


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
One of the first things u should do before using it, if you are using VS code or any other type of software, you should download opencv-python, mediapipe, and numpy.You can download them either from the python interpreter or from extentions or you can just download them by using ur terminal, but keep in mind that you have to use different commands depending on the different type of OS system you have.

Since I am currently using a MAC (m1) chip, I had some complications reaching the camera. So i had to make some changes, but the code overall, there is nothing to worry about I found a way to suppress the tensorflow logs. 

Keep in mind that your webcam can have a different integer for it, for me it was [0] but for you it can be 1,2 or maybe even 3!

for me , the hardest part was to access to the camera since it was the only thing i didnt know lol

after thatt i tried changing GBR into RGB since my camera really had some problems, but then i didnt need them.


Now for the Mediapipe modules, i used mp_hands to detect and hand landmark.  mp_face_mesh detects the face and tracks it. mp_drawing it draws landmarks on images.
To be utterly honest, i think my finger counting logic is not really accurate, its precise but not accurate towards it. i was planning to add negative numbers since when i showed 0 fingers it outputted as 1 so by adding negative numbers it would've showed 0. but it got really complicated so i left it as it is now.
I added small things like it shows the camera as if you are looking to a mirror(mirror effect) I made the camera recognize which hand is right or left. it alwasys quits with q. The last lines of the code are necessary or else the code would've been running for ever. 


lastly for where i got the information for facial expressions you can access the resource from;
        https://vhil.stanford.edu/sites/g/files/sbiybj29011/files/media/file/bailenson-evoked-emotion.pdf

you can mail me from 	rocketsci3939@gmail.com 


