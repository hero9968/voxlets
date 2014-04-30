#cd ../code/python/

import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import scipy.io
import sys


args = sys.argv

if len(args) < 4:
    print "Not enough args"
    sys.exit()

scene = args[1]
radius = float(args[2])
startIdx = int(args[3])
endIdx = int(args[4])


def loadOBJ(filename):
	numVerts = 0
	verts = []
	norms = []
	vertsOut = []
	normsOut = []
	k=0
	for line in open(filename, "r"):
		vals = line.split()
		if len(vals) == 0:
			continue
		elif vals[0] == "v":
			v = map(float, vals[1:4])
			verts.append(v)
		elif vals[0] == "vn":
			n = map(float, vals[1:4])
			norms.append(n)
		elif vals[0] == "f":
			for f in vals[1:]:
				#print f
				w = f.split()
				#print int(w[0])
				# OBJ Files are 1-indexed so we must subtract 1 below
				vertsOut.append(list(verts[int(w[0])-1]))
				numVerts += 1
	return vertsOut, verts
    

    
def drawMesh():
    glBegin(GL_TRIANGLES)
    for i in range(len(triangles)):
        glVertex3fv(triangles[i])
    glEnd()
    
def getPlane(mat):
    n = mat[0:3, 2]
    c = np.linalg.pinv(mat)[3,0:3]
    return [n, c]
    
def distPlane(n,p,q):
    return np.abs(np.dot(q-p,n))
    
def loadXform():
    global idx, transMatrix
    
    filename = "/Users/Michael/projects/shape_sharing/data/3D/basis_models/halo/mat_" + str(idx) + ".csv"    
   # xform = scipy.io.loadmat(`"/Users/Michael/Data/Others_data/google_warehouse/rotations/mat_" + str(idx) + ".mat")['H']
   
    # loading the transform from disk
    xform = np.genfromtxt(filename, delimiter=',')
    print "Xform is " + str(xform)
    
    # adjusting the radius according to user supplied arguments
    xform[0:3,3] = radius * xform[0:3,3]
    print "Xform after is " + str(xform)
	
	# taking the inverse
    transMatrix = np.linalg.pinv(xform.T)
    
    print "Xform " + str(idx) + " loaded."

zNear = 0.1
zFar = radius * 2

# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'
SPACE = '\040'

#global xform, transMatrix, startIdx, endIdx, idx, plane

#Width, Height = 1024, 1024
Width, Height = 640, 480

# Number of the glut window.
window = 0

modelsPath = "/Users/Michael/projects/shape_sharing/data/3D/basis_models/centred/"
savePath =   "/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/"
#mainPath = "/Users/Michael/Data/Derived/dino_chef_pc/models/"


idx = startIdx

transMatrix = np.zeros((4,4))

triangles, verts = loadOBJ(modelsPath + "/" + scene + ".obj")
#triangles, verts = loadOBJ("/Users/Michael/projects/shape_sharing/3D/model_render/data/cube.obj")

print("Loaded " + str(len(triangles)) + " triangles and " + str(len(verts)) + " verts")


# A general OpenGL initialization function.  Sets all of the initial parameters.
def InitGL(Width, Height):				# We call this right after our OpenGL window is created.
    glClearColor(0.0, 0.0, 0.0, 0.0)	# This Will Clear The Background Color To Black
    glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)				# The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
    glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
  
    
    # Calculate The Aspect Ratio Of The Window
    gluPerspective(43.0, float(Width)/float(Height), zNear, zFar)

  #  glMatrixMode(GL_MODELVIEW)
    glMatrixMode(GL_PROJECTION)


# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def ReSizeGLScene(Width, Height):
    if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small
	    Height = 1

    glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(43.0, float(Width)/float(Height), zNear, zFar)
    glMatrixMode(GL_MODELVIEW)
    
# The main drawing function.
def DrawGLScene():
	global transMatrix

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	# Clear The Screen And The Depth Buffer
	glLoadIdentity();					# Reset The View
    
	glLoadMatrixf(transMatrix)
	#gluLookAt()
    
    
	drawMesh()
    
	#  since this is double buffered, swap the buffers to display what just got drawn.
	glutSwapBuffers()
    
	return
    
def printDepth():
    global idx, transMatrix, verts
    depth = np.array(glReadPixels(0, 0, Width, Height, GL_DEPTH_COMPONENT, GL_FLOAT), dtype=np.float32)
    print "Depth size is " + str(depth.shape)
    
    depth.shape = Height, Width
    depth = np.flipud(depth)
    
    #linearize depth, from: http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    depth = 2.0 * zNear * zFar / (zFar + zNear - (2.0*depth-1.0) * (zFar - zNear))
    
    #get the point to plane distances
    plane = getPlane(transMatrix)
    dists = distPlane(plane[0], plane[1], verts)
    
    dico = dict(depth=depth, dists=dists)
    
    scipy.io.savemat(savePath + scene + "/depth_" + str(idx) + ".mat", dico)
    print "mat written"
    
    # seeing how many depth aren't at the maximum...
    print "Minimum depth is " + str(np.amin(depth))


# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
	# If escape is pressed, kill everything.
    if args[0] == ESCAPE:
        sys.exit()
        
    if args[0] == SPACE:
        printDepth()
   
def timerf(time):
    global idx
    
    print "timerfunc"
    
    if idx <= endIdx:
        loadXform()
        DrawGLScene()
        printDepth()
        print "Frame " + str(idx) + " done."
        idx += 1
        glutTimerFunc(10, timerf, 0)
    else:
        sys.exit()
        print "Done"
        
def main():
	global window, idx

	glutInit(sys.argv)

	# Select type of Display mode:
	#  Double buffer
	#  RGBA color
	# Alpha components supported
	# Depth buffer
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
	#glutInitDisplayMode(GLUT_DEPTH)

	# get a 640 x 480 window
	glutInitWindowSize(Width, Height)

	# the window starts at the upper left corner of the screen
	glutInitWindowPosition(0, 0)

	# Okay, like the C version we retain the window id to use when closing, but for those of you new
	# to Python (like myself), remember this assignment would make the variable local and not global
	# if it weren't for the global declaration at the start of main.
	window = glutCreateWindow("Jeff Molofee's GL Code Tutorial ... NeHe '99")

   	# Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
	# set the function pointer and invoke a function to actually register the callback, otherwise it
	# would be very much like the C version of the code.
	glutDisplayFunc(DrawGLScene)

	# Uncomment this line to get full screen.
	# glutFullScreen()

	# When we are doing nothing, redraw the scene.
	#glutIdleFunc(idleFunc)
    
	glutTimerFunc(10, timerf, 0)

	# Register the function called when our window is resized.
	glutReshapeFunc(ReSizeGLScene)
	# Register the function called when the keyboard is pressed.
	glutKeyboardFunc(keyPressed)

	# Initialize our window.
	InitGL(Width, Height)

	# Start Event Processing Engine
	glutMainLoop()
    
	#printDepth()

# Print message to console, and kick off the main to get it rolling.
main()