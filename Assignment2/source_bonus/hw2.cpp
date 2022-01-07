#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

# include <math.h>
# include <GL/glut.h>
# include <stdlib.h> //newly add

#define ROW 10
#define COLUMN 50 
#define MIN_LEN 10   //the range of length of logs    
#define MAX_LEN 20
#define LOG_THREAD 9 

int frogX = ROW;
int init_x; 
int frogY = (COLUMN-1) / 2;    //Initial coor of frog
int status = 0;   // "0" represent win, "1" represent lose, "2" represent quit
char value;
bool isOver = false;
int logLen[9];    //The length of each log
int logsLocation[9];
bool direction[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};  // "0" represent left. "1" represent right


float speed = 50000;  //speed
float mouseX, mouseY;
GLfloat los;

pthread_mutex_t mutex;
pthread_cond_t threshold_cv;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog;

char map[ROW+10][COLUMN] ; 

//Draw the terminal information
void termPrint(float x, float y, char *text) {
	glClearColor(0.7f, 0.7f, 0.7f, 0.7f);
	glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(x,y);
    for (int i=0; text[i]; i++){
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);
    }
	glutSwapBuffers();
}


//Draw the window
void display(int nTimerID) {
	float x_init = -1.0f;
	float y_init = -0.5f;
	float x_len = 0.042f;
	float y_len1 = 0.14f;
	float y_len2 = 0.04f;
	float y_len3 = 0.1f;
	float PI = 3.141593;
	int frog_x, frog_y;

	char str[20];
	if (isOver) {
		sleep(2);
		pthread_exit(NULL);
	} else if (value == ' ' || frog.y < 1 || frog.y > 47) {
		if (init_x != 0 && init_x != 10) {
			map[init_x][frog.y] = '=';
		} 
		status = 1;
		isOver = true;
		sprintf(str, "You lose the game!!");
		termPrint(-0.3, 0.2,str);
	} else if (frog.x == 0) {
		status = 0;
		isOver = true;
		sprintf(str, "You win the game!!");
		termPrint(-0.3, 0.2,str);
	} else if (status == 2) {
		isOver = true;
		sprintf(str, "You exit the game!!");
		termPrint(-0.3, 0.2,str);
	} else {

		glClearColor(0.6f, 0.6f, 0.6f, 0.6f);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

		//Draw the boundaries
		glBegin(GL_QUADS);
		glColor3f(0.7f, 0.7f, 0.7f);
		glVertex2f(-1.0f, 0.95f);
		glVertex2f(1.0f, 0.95f);
		glVertex2f(-1.0f, -0.45f);
		glVertex2f(1.0f, -0.45f);
		glEnd();

		glLineWidth(14.0f);
		glBegin(GL_LINES);
		glColor3f(0.9f, 0.9f, 0.9f);
		for (int i = 1; i < COLUMN - 1; i++) {
			GLfloat x_los = x_init + i * x_len;
			glVertex2f(x_los, 1.0f);
			glVertex2f(x_los, 0.9f);
		}

		for (int i = 1; i < COLUMN - 1; i++) {
			GLfloat x_los = x_init + i * x_len;
			glVertex2f(x_los, -0.5f);
			glVertex2f(x_los, -0.4f);
		}

		//Draw the frog
		glColor3f(0.0f, 0.8f, 0.2f);
		for (int i = 0; i <= ROW; i++) {
			for (int j = 0; j < COLUMN - 1; j++) {
				if (map[i][j] == '0') {
					frog_x = i;
					frog_y = j;
					
					GLfloat x_los = x_init + j * x_len;
					GLfloat y_los1 = y_init + (ROW - i + 1) * y_len1;
					GLfloat y_los2 = y_init + (ROW - i + 1) * y_len1 - y_len3;
					GLfloat y_los3 = y_init + (ROW - i + 1) * y_len1 - y_len2 - y_len3;
					glVertex2f(x_los, y_los1 );
					glVertex2f(x_los, y_los2 );
					glColor3f(0.0f, 0.0f, 0.0f);
					glVertex2f(x_los, y_los2 );
					glVertex2f(x_los, y_los2 );
					if (i == ROW) {
						glColor3f(0.8f, 0.8f, 0.8f);
						glVertex2f(x_los, y_los2 );
						glVertex2f(x_los, y_los3 );
					} else {
						glColor3f(0.2f, 0.0f, 0.0f);
						glVertex2f(x_los, y_los2 );
						glVertex2f(x_los, y_los3 );
					}
				}
			}
		}

		//Draw the logs
		glColor3f(0.2f, 0.0f, 0.0f);
		for (int i = 1; i <= ROW; i++) {
			for (int j = 1; j < COLUMN ; j++) {
				if (map[i][j] == '=') {
					GLfloat x_los = x_init + j * x_len;
					GLfloat y_los1 = y_init + (ROW - i ) * y_len1;
					GLfloat y_los2 = y_init + (ROW - i) * y_len1 + y_len3;
					glVertex2f(x_los, y_los1 );
					glVertex2f(x_los, y_los2 );
				}
			}
		}
		glEnd();

		//Draw the slide bar
		glLineWidth(5.0f);
		glBegin(GL_LINES);
		glColor3f(1.0f, 1.0f, 1.0f);
		glVertex2f(-0.5f, -0.7f);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex2f(0.5f, -0.7f);
		glEnd();

		GLint circle_points = 100;
		glBegin(GL_POLYGON);
		for (int i = 0; i < circle_points; i++) {
			GLfloat angle = 2*PI*i/circle_points;
			glColor3f(0.0f, 0.0f, 0.2f);
			glVertex2f(cos(angle) / 45 + los, sin(angle) / 30 - 0.7);
		}
		glColor3f(0.9f, 0.2f, 0.2f);
		glEnd();

		glutSwapBuffers();
	}

	glutTimerFunc(10, display, 1);
}


//Mouse monitor
void onMouseTap(int button, int state, int x, int y ) {
	mouseX = x;
    mouseY = y;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		if ( mouseX >= 150 && mouseX <= 450 &&  mouseY >= 332 && mouseY <= 348) {
			los = (GLfloat)(-0.5f + (mouseX - 150) / 300);
			speed = 50000 - ((mouseX - 300) * 30000) / 150;
			glutPostRedisplay();
		}
    }
}


//Keyboard monitor
void onKeyTap(unsigned char key, int x, int y) {
	init_x = frog.x;
	if (frog.x == 0 || frog.x == 10) {
		map[frog.x][frog.y] = '|'; 
	}

	if( (key == 'w' || key == 'W') && (frog.x >= 1) )
		frog.x--;	
	if( (key == 'a' || key == 'A') && (frog.y >= 1) )
		frog.y--;
	if( (key == 'd' || key == 'D') && (frog.y <= 47) )
		frog.y++;
	if( (key == 's' || key == 'S') && (frog.x <= 9) )
		frog.x++;
	if( key == 'q' || key == 'Q' ){
		status = 2;
	}

	value = map[frog.x][frog.y];

	glutPostRedisplay();
}

//Control the move of logs (nine threads)
void *logs_move( void *t ){
	/*  Move the logs  */
	long row;
	row = (long)t + 1;
	while (!isOver) {
		pthread_mutex_lock(&mutex);
		if (direction[row-1]) {
			for (int i = 0; i < logLen[row-1] + 1; i++) {
				map[row][(logsLocation[row-1] + i) % (COLUMN - 1)] = '=' ;
			}
			map[row][logsLocation[row-1] % (COLUMN - 1)] = ' ';
		} else {
			for (int i = 0; i < logLen[row-1] + 1; i++) {
				if ((logsLocation[row-1] - i) % (COLUMN - 1) >= 0 ) {
					map[row][(logsLocation[row-1] - i) % (COLUMN - 1)] = '=' ;
				} else {
					map[row][COLUMN - abs(logsLocation[row-1] - i) % (COLUMN - 1) - 2] = '=' ;
				}
			}
			if ((logsLocation[row-1]) % (COLUMN - 1) >= 0 ) {
				map[row][(logsLocation[row-1]) % (COLUMN - 1)] = ' ' ;
			} else {
				map[row][COLUMN - abs(logsLocation[row-1]) % (COLUMN - 1) - 2] = ' ';
			}
		}

		if (direction[row-1]) {
			logsLocation[row-1]++;
		} else {
			logsLocation[row-1]--;
		}

		if (map[frog.x][frog.y] == '=') {
			if (direction[frog.x-1]) {
				frog.y++;
			} else {
				frog.y--;
			}
		}
		map[frog.x][frog.y] = '0';

		pthread_mutex_unlock(&mutex);
		usleep(speed);
	}

	pthread_exit(NULL);
}


//Draw the GUI window
void *init_window( void *t ) {
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);   	//Initialize the GLUT window

    glutInitWindowSize (600, 400);     //Determine the size and location of the window
    glutInitWindowPosition (400, 200);

    glutCreateWindow ("Jared's Frog Game");   //the name of the window

	glutKeyboardFunc( onKeyTap );   //Keyboard monitor
	glutMouseFunc( onMouseTap );   //Mouse monitor
	glutTimerFunc(10, display, 1);

    glutMainLoop( );
}


int main( int argc, char *argv[] ){
	glutInit (&argc, argv);

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ); 
	map[frog.x][frog.y] = '0' ; 

	/* Initialize the status of logs (including the length and initial position)：*/
	srand((unsigned)time(0));  
	for (int k = 0; k < 9; k++) {
		logLen[k] = (rand() % (MAX_LEN - MIN_LEN + 1))+ MIN_LEN; 
	}
	for (int k = 0; k < 9; k++) {
		logsLocation[k] = rand() % 49;
	}


	/*  Create pthreads for wood move and frog control.  */	
	pthread_t threads[LOG_THREAD];
	pthread_t window_tid;
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init (&threshold_cv, NULL);

	long id;
	for (id = 0; id < LOG_THREAD; id++) {
		pthread_create(&threads[id], NULL, logs_move, (void *)id);
	}
	pthread_create(&window_tid, NULL, init_window, NULL);

	for (id = 0; id < LOG_THREAD; id++) {
		pthread_join(threads[id], NULL);
	}
	pthread_join(window_tid, NULL);

	/*  Display the output for user: win, lose or quit.  */
	printf("\033[H\033[2J");
	if (status == 0) {
		printf("You win the game!!\n");
	} else if (status == 1) {
		printf("You lose the game!!\n");
	} else if (status == 2) {
		printf("You exit the game!!\n");
	}

	pthread_cond_destroy(&threshold_cv);
	pthread_mutex_destroy(&mutex);
    pthread_exit(NULL);

	return 0;
}
