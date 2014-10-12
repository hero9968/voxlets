#cd ../code/python/

import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import scipy.io
import sys
from plyfile import PlyData, PlyElement
import timeit
import h5py

bigbird_folder = "/Users/Michael/projects/shape_sharing/data/bigbird/"
mesh_folder = "/Users/Michael/projects/shape_sharing/data/bigbird_meshes/"

args = sys.argv

if len(args) < 2:
    print "Not enough args"
    sys.exit()

modelname = args[1]

def load_all_poses():
    filename = bigbird_folder + "poses_to_use.txt"
    return [pose.strip() for pose in open(filename, 'r')]

poses = load_all_poses()
endIdx = len(poses)

global showallverts


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
    #import pdb; pdb.set_trace()
    return vertsOut, verts
    


def drawMesh():
    global showallverts
    glCallList(showallverts)


def getPlane(mat):
    n = mat[0:3, 2]
    c = np.linalg.pinv(mat)[3,0:3]
    return [n, c]


def distPlane(n,p,q):
    return np.abs(np.dot(q-p,n))


def loadXform():
    global idx, transMatrix
    this_pose = poses[idx]
    print this_pose

    cameraname, pose_angle = this_pose.split("_")

    # loading the transform from NP5 to the camera
    print "Loading " + mesh_folder + modelname + "/calibration.h5"
    calib_h5 = h5py.File(mesh_folder + modelname + "/calibration.h5", 'r')
    H_CAM_ir_from_NP5 = np.array(calib_h5['H_' + cameraname + '_ir_from_NP5'])

    # loading the transform from the mesh to np5
    np5_pose_filename = mesh_folder + modelname + "/poses/NP5_" + pose_angle + "_pose.h5"
    print "Loading " + np5_pose_filename
    pose_h5 = h5py.File(np5_pose_filename, 'r')
    mesh_to_np5 = np.linalg.pinv(np.array(pose_h5['H_table_from_reference_camera']))

    # the extrinsic matrix which works in the manual projection
    inv_extrinsic = H_CAM_ir_from_NP5.dot(mesh_to_np5)
    #extrinsic = H_CAM_ir_from_NP5.dot(np.linalg.pinv(H_table_from_reference_camera))
    #print inv_extrinsic
    #inv_extrinsic[:3, 3] = np.dot(inv_extrinsic[:3,:3], inv_extrinsic[:3, 3])
    # forming the final transformation
    xform = inv_extrinsic.T
    xform[3, 2] = -xform[3, 2]

    #print H_CAM_ir_from_NP5
    #print H_table_from_reference_camera
    #print xform.T

    #xform[:3,3] = np.dot(xform[:3, :3], xform[:3,3])

    #print xform

    transMatrix = xform #np.linalg.pinv(xform.T)


zNear = 0.1
zFar = 10.0 # I had this set to 2 * radius for a bit...

# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'
SPACE = '\040'

#Width, Height = 320, 240
Width, Height = 640, 480
y_angle = 43.0

# Number of the glut window.
window = 0

meshPath = "/Users/Michael/projects/shape_sharing/data/bigbird_meshes/" + modelname + "/meshes/poisson.obj"
triangles, verts = loadOBJ(meshPath)

savePath =   "/Users/Michael/projects/shape_sharing/data/bigbird_renders/render_backface/" + modelname + "/"

idx = 0

transMatrix = np.zeros((4,4))


# creating a display list for putting the triangles on the screen
#showallverts = glGenLists(1)
#glNewList(showallverts, GL_COMPILE)
#glBegin(GL_TRIANGLES)
#for i in range(len(triangles)):
#   print "s"
   #glVertex3fv(triangles[i])
#glEnd()
#glEndList()

#sys.exit("done here")
 
print("Loaded " + str(len(triangles)) + " triangles and " + str(len(verts)) + " verts")


# A general OpenGL initialization function.  Sets all of the initial parameters.
def InitGL(Width, Height):              # We call this right after our OpenGL window is created.
    glClearColor(0.0, 0.0, 0.0, 0.0)    # This Will Clear The Background Color To Black
    glClearDepth(0.0)                   # Enables Clearing Of The Depth Buffer
    #glDepthFunc(GL_LESS)                # The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)             # Enables Depth Testing
    glDisable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)                # Enables Smooth Color Shading

    glDepthFunc(GL_GREATER)             # The Type Of Depth Test To Do
    glDepthMask(GL_TRUE)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
  
    # Calculate The Aspect Ratio Of The Window
    gluPerspective(y_angle, float(Width)/float(Height), zNear, zFar)

  #  glMatrixMode(GL_MODELVIEW)
    glMatrixMode(GL_PROJECTION)


# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def ReSizeGLScene(Width, Height):

    glViewport(0, 0, Width, Height)     # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(y_angle, float(Width)/float(Height), zNear, zFar)
    glMatrixMode(GL_MODELVIEW)
    
# The main drawing function.
def DrawGLScene():
    global transMatrix

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); # Clear The Screen And The Depth Buffer
    glLoadIdentity();                   # Reset The View

    glLoadMatrixf(transMatrix)
    #gluLookAt(915.252+100, 48.7601+100, 76.333+100, 915.252, 48.7601, 76.333, 0, 1.0, 1.0)

    drawMesh()
    
    #  since this is double buffered, swap the buffers to display what just got drawn.
    glutSwapBuffers()
    
    return
    
def printDepth():
    global idx, transMatrix, verts
    
    tempdepth = glReadPixels(0, 0, Width, Height, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth = np.array(tempdepth, dtype=np.float32)

    depth.shape = Height, Width
    depth = np.flipud(depth)
    
    #linearize depth, from: http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    depth = 2.0 * zNear * zFar / (zFar + zNear - (2.0*depth-1.0) * (zFar - zNear))
    
    #get the point to plane distances
    plane = getPlane(transMatrix)
    dists = distPlane(plane[0], plane[1], verts)
    
    dico = dict(depth=depth, dists=dists)
    scipy.io.savemat(savePath + "/" + poses[idx] + "_renderbackface.mat", dico)
    #scipy.io.savemat("test_out.mat", dico)


# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    # If escape is pressed, kill everything.
    if args[0] == ESCAPE:
        sys.exit()
        
    if args[0] == SPACE:
        printDepth()


def timerf(time):
    global idx, meshmodel
        
    if idx < endIdx:
        tic = timeit.default_timer()
        loadXform()
        DrawGLScene()
        printDepth()
        idx += 1
        glutTimerFunc(10, timerf, 0)
    else:
        sys.exit()
        print "Done"


def main():
    global window, idx, showallverts

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

    # setting up the display list
    showallverts = glGenLists(1)
    glNewList(showallverts, GL_COMPILE)
    # every three vertices do glben 
    
    glBegin(GL_TRIANGLES)
    #for i in range(0, len(triangles)-4, 4):
    for i in range(0, len(triangles)):
        glVertex3fv(triangles[i])
    glEnd()
    glEndList()

    # Start Event Processing Engine
    glutMainLoop()
        
    #printDepth()

# Print message to console, and kick off the main to get it rolling.
tic = timeit.default_timer()
main()
print 'Final time: ' + str(timeit.default_timer() - tic)

