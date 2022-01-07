#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	/* fork a child process */
    int status;
	pid_t pid;
	printf("Process start to fork\n");
    pid = fork();

	if(pid==-1){
        perror("fork");
        exit(1);
    }
    else{
        /* Child process */
        if(pid==0){
            int i, j;
            char *arg[argc];

            printf("I'm the Child Process, my pid = %d\n", getpid());

            for(i=0 ; i<argc-1 ;i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1]=NULL;

	        /* execute test program */ 
            printf("Child process start to execute test program:\n");
            execve(arg[0],arg,NULL);

            perror("execve");
            exit(EXIT_FAILURE);
        }
        /* Parent process */
        else{
            printf("I'm the Parent Process, my pid = %d\n", getpid());
            
	        /* wait for child process terminates */
            waitpid(pid, &status, WUNTRACED);  //Reports child process' signal(also stop)
			printf("Parent process receiving the SIGCHILD signal!\n");

            /* check child process' termination status */
            if(WIFEXITED(status)){   //normal termination
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)){
                if(WTERMSIG(status) == 6){   //abort
                    printf("child process get SIGABRT signal\n");
                    printf("child process aborted\n");
                }
                else if(WTERMSIG(status) == 14){   //alarm
                    printf("child process get SIGALRM signal\n");
                    printf("child process alarmed\n");
                }
                else if(WTERMSIG(status) == 7){   //bus
                    printf("child process get SIGBUS signal\n");
                    printf("child process has bus error\n");
                }
                else if(WTERMSIG(status) == 8){   //floating
                    printf("child process get SIGFPE signal\n");
                    printf("child process has floating error\n");
                }
                else if(WTERMSIG(status) == 1){   //hangup
                    printf("child process get SIGHUP signal\n");
                    printf("child process is hang up\n");
                }
                else if(WTERMSIG(status) == 4){   //illegal_instr
                    printf("child process get SIGILL signal\n");
                    printf("child process has illegal_instr\n");
                }
                else if(WTERMSIG(status) == 2){   //interrupt
                    printf("child process get SIGINT signal\n");
                    printf("child process interrupted\n");
                }
                else if(WTERMSIG(status) == 9){   //kill
                    printf("child process get SIGKILL signal\n");
                    printf("child process is killed\n");
                }
                else if(WTERMSIG(status) == 13){   //pipe
                    printf("child process get SIGPIPE signal\n");
                    printf("child process piped\n");
                }
                else if(WTERMSIG(status) == 3){    //quit
                    printf("child process get SIGQUIT signal\n");
                    printf("child process quitted\n");
                }
                else if(WTERMSIG(status) == 11){   //segement_fault
                    printf("child process get SIGSEGV signal\n");
                    printf("child process has segment_fault\n");
                }
                else if(WTERMSIG(status) == 15){   //terminate
                    printf("child process get SIGTERM signal\n");
                    printf("child process terminated\n");
                }
                else if(WTERMSIG(status) == 5){   //trap
                    printf("child process get SIGTRAP signal\n");
                    printf("child process is trapped\n");
                }
                else{
                    printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
                }
                printf("CHILD EXECUTION FAILED\n");
            }
            else if(WIFSTOPPED(status)){   //stop
                printf("child process get SIGSTOP signal\n");
                printf("child process stopped\n");
                printf("CHILD PROCESS STOPPED\n");
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }

            exit(1);
        }
    }
	
    return 0;
	
}
