#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sys/wait.h>

/* Using recursion to create child process */
void create_process(int argc, int num, char *arg[], int *pidId[], int *pidStatus[]){
	int status;
	pid_t pid;
	pid = fork();

	if(pid==-1){
		perror("fork");
		exit(1);
	} else {
		/* child process */
		if (pid==0) {
			//printf("This is child process: %d\n", getpid());
			if (argc-1 > num + 1) {
				create_process(argc, num + 1, arg, pidId, pidStatus);  // recursion
			}
			*pidId[num+1] = getpid();
			execve(arg[num],arg,NULL);
		} 
		/* parent process */
		else {
			//printf("This is parent process: %d\n", getpid());
			*pidId[num] = getpid();
			waitpid(pid, &status, 0);

			if(WIFEXITED(status)){
				*pidStatus[num] = 0;    //using status 0 to represent normally exit
			} else if(WIFSIGNALED(status)){
				*pidStatus[num] = WTERMSIG(status);
			}
		}
	}
}


int main(int argc,char *argv[]){

	int i, j, k, l, n;
    char *arg[argc];
	int *pidId[argc];  //store all process IDs
	int *pidStatus[argc];  //store all terminated statuses

	for(i=0 ; i<argc-1 ;i++){
        arg[i]=argv[i+1];
    }
	arg[argc-1]=NULL;

	/* create mapping area */
	for(j=0 ; j<argc ;j++) {
		pidId[j] = (int*)mmap(NULL, 40, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
		pidStatus[j] = (int*)mmap(NULL, 40, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
		if (pidId[j] == MAP_FAILED) {
			perror("mmap error");
			exit(1);
		}
		if (pidStatus[j] == MAP_FAILED) {
			perror("mmap error");
			exit(1);
		}
	}	

	/* create child process */
	create_process(argc, 0, arg, pidId, pidStatus);
	
	/* Print out the process tree */
	printf("\n---\n");
	printf("Process tree: ");
	for(l=0 ; l<argc ;l++) {
		printf("%d", *pidId[l]);
		if (l != argc-1) {
			printf("->");
		}
	}
	printf("\n");

	for(l=0 ; l < argc-1 ;l++) {
		printf("Child process %d of parent process %d ", *pidId[argc-l-1], *pidId[argc-l-2]);
		if (*pidStatus[argc-l-2] == 0) {
			printf("terminated normally with exit code 0\n");
		} else {
			int temp = *pidStatus[argc-l-2];
			printf("is terminated by signal %d ", temp);
			if(temp == 6){
                printf("(Abort)\n");
            } else if (temp == 14){
                printf("(Alarm)\n");
            } else if (temp == 7){
                printf("(Bus)\n");
            } else if (temp == 8){
                printf("(Floating)\n");
            } else if (temp == 1){
                printf("(Hangup)\n");
            } else if (temp == 4){
                printf("(Illegal_instr)\n");
            } else if (temp == 2){
                printf("(Interrupt)\n");
            } else if (temp == 9){
                printf("(Kill)\n");
            } else if (temp == 13){
                printf("(Pipe)\n");
            } else if (temp == 3){
                printf("(Quit)\n");
            } else if (temp == 11){
                printf("(Segment_fault)\n");
            } else if (temp == 15){
                printf("(Terminate)\n");
            } else if (temp == 5){
                printf("(Trap)\n");
            }
		}
	}
	printf("Myfork process (%d) terminated normally\n", *pidId[0]);

 
	/*release mapping area */
	for(k=0 ; k<argc ;k++) {
		int ret1 = munmap(pidId[k], 40);
		int ret2 = munmap(pidStatus[k], 40);
		if(ret1 == -1) {
			perror("munmap error");
			exit(1);
		}
		if(ret2 == -1) {
			perror("munmap error");
			exit(1);
		}
	}
	return 0;
}
