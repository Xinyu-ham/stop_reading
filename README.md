# stop_reading
A very useful tool to help people get rid of a very disgusting habit - reading.

This bot tries to track where on screen your eyes are focused on, and blurs out the area so you won't be able to read, and hence protecting your eyes.

![Alt text](example.png?raw=true "Example Screenshot")

1. Use Google's mediapipe to obtain a map of facial landmarks
2. calculate the eye angle based on the relative position of the eye and the pupil 
3. Map angle to screen position through calibrating by starting at the opposite corners of the screen. 
4. Blur out whereever the eyes are focusing on the screen
