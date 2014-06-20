#include <iostream>

#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
using namespace std;

void display()
{
    int w = glutGet( GLUT_WINDOW_WIDTH );
    int h = glutGet( GLUT_WINDOW_HEIGHT );

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    double ar = w / static_cast< double >( h );
    gluPerspective( 60, ar, 0.1, 10 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0, 0, -4 );

    static float angle = 0;
    angle += 3;

    glPushMatrix();
    glRotatef( angle, 0.1, 0.5, 0.3 );
    glColor3ub( 255, 0, 0 );
        
    glPopMatrix();

    vector< GLfloat > depth( w * h, 0 );
    glReadPixels( 0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0] ); 

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

        float sumof_depth = 0;
    for (size_t i = 0; i < w*h; ++i)
        sumof_depth += depth[i];
    std::cout << "Sum of depths is " << sumof_depth << std::endl;

}

void timer( int value )
{
    glutPostRedisplay();
    glutTimerFunc( 16, timer, 0 );
}

int main( int argc, char **argv )
{
    glutInit( &argc, argv );
    //glutInitDisplayMode( GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize( 600, 600 );
    glutCreateWindow( "GLUT" );
    glewInit();
    glutDisplayFunc( display );
    glutTimerFunc( 0, timer, 0 );
    glEnable( GL_DEPTH_TEST );
    glutMainLoop();
    return 0;
}