// From http://stackoverflow.com/questions/24266815/render-the-depth-buffer-in-opengl-without-shaders
#include <stdlib.h>
#include <iostream>

#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>

using namespace std;

GLuint car;

struct myVertex{
    float x;
    float y;
    float z;
};

void loadObj(char *fname)
{
    FILE *fp;
    int read;
    GLfloat x, y, z;
    char ch;
    car=glGenLists(1);
    fp=fopen(fname,"r");
    if (!fp)
    {
        printf("can't open file %s\n", fname);
        exit(1);
    }
    glPointSize(2.0);

    glNewList(car, GL_COMPILE);
    {
        //glPushMatrix();
        glBegin(GL_TRIANGLES);
        vector<myVertex> verts;
        while(!(feof(fp)))
        {
            read=fscanf(fp,"%c %f %f %f",&ch,&x,&y,&z);
            myVertex vert;
            if(read==4&&ch=='v')
            {
                vert.x = x;
                vert.y = y;
                vert.z = z;
                verts.push_back(vert);
            }
            if(read==4&&ch=='f')
            {
                int f1 = (int)x;
                int f2 = (int)y;
                int f3 = (int)z;
                //cout << "SIze of verts is " << verts.size() << endl;
                //cout << f1 << " , " << f2 << " , " << f3 << endl;
                glColor3f(255, 255, 0);
                //glVertex3f(0, 0, 1);
                //glVertex3f(0, 1, 1);
                //glVertex3f(10, 0, 0);
                glVertex3f(verts.at(f1-1).x, verts.at(f1-1).y, verts.at(f1-1).z);
                glVertex3f(verts.at(f2-1).x, verts.at(f2-1).y, verts.at(f2-1).z);
                glVertex3f(verts.at(f3-1).x, verts.at(f3-1).y, verts.at(f3-1).z);
            }
        }

        glEnd();
    }
    //glPopMatrix();
    glEndList();
    fclose(fp);
}
//.obj loader code ends here

void display()
{
    int w = glutGet( GLUT_WINDOW_WIDTH );
    int h = glutGet( GLUT_WINDOW_HEIGHT );

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    double ar = w / static_cast< double >( h );
    const float zNear = 0.1;
    const float zFar = 10.0;
    gluPerspective( 60, ar, zNear, zFar );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0, 0, -4 );

    static float angle = 0;
    angle += 3;

    glPushMatrix();
    glRotatef( angle, 0.1, 0.5, 0.3 );
    glColor3ub( 255, 0, 0 );
    glutSolidTeapot( 1 );
    //loadObj("/Users/Michael/projects/shape_sharing/data/3D/basis_models/databaseFull/models/8508808961d5a0b2b1f2a89349f43b2.obj");
    //loadObj("./tea.obj");
    glPopMatrix();

    vector< GLfloat > depth( w * h, 0 );
    glReadPixels( 0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0] ); 

    // linearize depth
    // http://www.geeks3d.com/20091216/geexlab-how-to-visualize-the-depth-buffer-in-glsl/
    for( size_t i = 0; i < depth.size(); ++i )
    {
        depth[i] = ( 2.0 * zNear ) / ( zFar + zNear - depth[i] * ( zFar - zNear ) );
    }

    static GLuint tex = 0;
    if( tex > 0 )
        glDeleteTextures( 1, &tex );
    glGenTextures(1, &tex);
    glBindTexture( GL_TEXTURE_2D, tex);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_FLOAT, &depth[0] );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, w, 0, h, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glEnable( GL_TEXTURE_2D );
    glColor3ub( 255, 255, 255 );
    glScalef( 0.3, 0.3, 1 );
    glBegin( GL_QUADS );
    glTexCoord2i( 0, 0 );
    glVertex2i( 0, 0 );
    glTexCoord2i( 1, 0 );
    glVertex2i( w, 0 );
    glTexCoord2i( 1, 1 );
    glVertex2i( w, h);
    glTexCoord2i( 0, 1 );
    glVertex2i( 0, h );
    glEnd();
    glDisable( GL_TEXTURE_2D );

    glutSwapBuffers();
}

void timer( int value )
{
    glutPostRedisplay();
    glutTimerFunc( 16, timer, 0 );
}

int main( int argc, char **argv )
{
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( 600, 600 );
    glutCreateWindow( "GLUT" );
    glewInit();
    glutDisplayFunc( display );
    glutTimerFunc( 0, timer, 0 );
    glEnable( GL_DEPTH_TEST );
    glutMainLoop();
    return 0;
}