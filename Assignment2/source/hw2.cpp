#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define MIN_LEN 10   //the range of length of logs    
#define MAX_LEN 20
#define LOG_THREAD 9 

int frogX = ROW;
int frogY = (COLUMN-1) / 2;    //Initial coor of frog
int status = 0;   // "0" represent win, "1" represent lose, "2" represent quit
bool isOver = false;
int logLen[9];    //The length of each log
int logsLocation[9];
bool direction[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};  // "0" represent left. "1" represent right
pthread_mutex_t mutex;
pthread_cond_t threshold_cv;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog;
char map[ROW+10][COLUMN] ; 


// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
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
					map[row][COLUMN - abs(logsLocation[row-1] - i) % (COLUMN - 1) - 1] = '=' ;
				}
			}
			if ((logsLocation[row-1]) % (COLUMN - 1) >= 0 ) {
				map[row][(logsLocation[row-1]) % (COLUMN - 1)] = ' ' ;
			} else {
				map[row][COLUMN - abs(logsLocation[row-1]) % (COLUMN - 1) - 1] = ' ';
			}
		}

		if (direction[row-1]) {
			logsLocation[row-1]++;
		} else {
			logsLocation[row-1]--;
		}

		pthread_cond_signal(&threshold_cv);
		pthread_mutex_unlock(&mutex);
		usleep(50000);
	}

	pthread_exit(NULL);
}


//Control the move of frog (one thread)
void *frog_move( void *t ){
	/*  Check keyboard hits, to change frog's position or quit the game. */
	while (!isOver) {
		pthread_mutex_lock(&mutex);
		pthread_cond_wait(&threshold_cv, &mutex);

		int init_x = frog.x;
		if (frog.x == 0 || frog.x == 10) {
			map[frog.x][frog.y] = '|'; 
		}

		if(kbhit()){
			char dir = getchar() ;

			if( (dir == 'w' || dir == 'W') && (frog.x >= 1) )
				frog.x--;	
			if( (dir == 'a' || dir == 'A') && (frog.y >= 1) )
				frog.y--;
			if( (dir == 'd' || dir == 'D') && (frog.y <= 47) )
				frog.y++;
			if( (dir == 's' || dir == 'S') && (frog.x <= 9) )
				frog.x++;
			if( dir == 'q' || dir == 'Q' ){
				status = 2;
				isOver = true;
			}
		}

		if (map[frog.x][frog.y] == '=') {
			if (direction[frog.x-1]) {
				frog.y++;
			} else {
				frog.y--;
			}
		}

		/*  Check game's status  */
		if (map[frog.x][frog.y] == ' ' || (frog.x != 10 && frog.y < 1) || (frog.x != 10 && frog.y > 47)) {
			if (init_x != 0 && init_x != 10) {
				map[init_x][frog.y] = '=';
			} 
			status = 1;
			isOver = true;
		} else if (frog.x == 0) {
			status = 0;
			isOver = true;
		}
		map[frog.x][frog.y] = '0';

		/*  Print the map on the screen  */
		printf("\033[H\033[2J");

		for (int j = 0; j <= ROW; ++j) {
			puts( map[j] );
		}
		pthread_mutex_unlock(&mutex);
	}

	pthread_exit(NULL);
}


int main( int argc, char *argv[] ){
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

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );


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
	pthread_t frog_tid;
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init (&threshold_cv, NULL);

	long id;
	for (id = 0; id < LOG_THREAD; id++) {
		pthread_create(&threads[id], NULL, logs_move, (void *)id);
	}
	pthread_create(&frog_tid, NULL, frog_move, NULL);

	for (id = 0; id < LOG_THREAD; id++) {
		pthread_join(threads[id], NULL);
	}
	pthread_join(frog_tid, NULL);

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
