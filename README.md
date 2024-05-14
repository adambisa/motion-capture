# Motion Capture Script

This script captures (almost) any movement caught by camera and stores it in a .csv file using the **pandas** module  (currently hard coded as times.csv) with start and end time for the movement.

I do this by capturing video using the **open-cv** library I then find the grayscale of the video input, and use Gaussian Blur and find the delta treshhold  and draw find contours
on the **treshdelta** image and then draw a rectangle on the same coordinates in the original.

I am able to find movement by reflecting light which shows as ligher - whiter pixels which I can track. 

As of now it can track one moving object closest to the camera.

## first run : pip install -r requirements.txt
