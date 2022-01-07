#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;

struct wait_opts wo;

struct wait_opts { 
	enum pid_type wo_type; 
	int wo_flags; 
	struct pid *wo_pid;
	struct siginfo __user *wo_info;
	int __user *wo_stat; 
	struct rusage __user *wo_rusage; 
	wait_queue_t child_wait; 
	int notask_error;
};

extern long _do_fork(unsigned long clone_flags,
	    	unsigned long stack_start,
	    	unsigned long stack_size,
	    	int __user *parent_tidptr,
	    	int __user *child_tidptr,
	    	unsigned long tls);
extern int do_execve(struct filename *filename,
			const char __user *const __user *__argv,
			const char __user *const __user *__envp);
extern struct filename * getname(const char __user * filename);
extern long do_wait(struct wait_opts *wo);


//implement exec function
int my_exec(void) {
	int result;
	const char path[] = "/opt/test";
	const char *const argv[]={path, NULL, NULL};
	const char *const envp[]={"HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL};

	struct filename * my_filename = getname(path);

	printk("[program2] : child process\n");

	/* execute a test program in child process */
	result=do_execve(my_filename, argv, envp);

	/* if exec succeeded */
	if (!result) 
		return 0;

	/* if exec failed */
	do_exit(result);
}

//implement wait function
void my_wait(pid_t pid) {
	int status;
	int a;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED|WUNTRACED;    //include the case of stop
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*)&status;
	wo.wo_rusage = NULL;

	/* wait until child process terminates */
	a=do_wait(&wo);

	/* print out the child process info */
	if (!status) {
		printk("[program2] : Normal termination");
		printk("[program2] : child process normally terminated");
		printk("[program2] : The return signal is 0");
	} else {
		if (status == 135) {
			printk("[program2] : get SIGBUS signal");
			printk("[program2] : child process has bus error");
			printk("[program2] : The return signal is 7");
		} else if (status == 134) {
			printk("[program2] : get SIGABRT signal");
			printk("[program2] : child process aborted");
			printk("[program2] : The return signal is 6");
		} else if (status == 14) {
			printk("[program2] : get SIGALRM signal");
			printk("[program2] : child process alarmed");
			printk("[program2] : The return signal is 14");
		} else if (status == 136) {
			printk("[program2] : get SIGFPE signal");
			printk("[program2] : child process has floating error");
			printk("[program2] : The return signal is 8");
		} else if (status == 1) {
			printk("[program2] : get SIGHUP signal");
			printk("[program2] : child process HANGUPED");
			printk("[program2] : The return signal is 1");
		} else if (status == 132) {
			printk("[program2] : get SIGILL signal");
			printk("[program2] : child process has illegal_instr");
			printk("[program2] : The return signal is 4");
		} else if (status == 2) {
			printk("[program2] : get SIGINT signal");
			printk("[program2] : child process interrupted");
			printk("[program2] : The return signal is 2");
		} else if (status == 9) {
			printk("[program2] : get SIGKILL signal");
			printk("[program2] : child process killed");
			printk("[program2] : The return signal is 9");
		} else if (status == 13) {
			printk("[program2] : get SIGPIPE signal");
			printk("[program2] : child process piped");
			printk("[program2] : The return signal is 13");
		} else if (status == 131) {
			printk("[program2] : get SIGQUIT signal");
			printk("[program2] : child process quitted");
			printk("[program2] : The return signal is 3");
		} else if (status == 139) {
			printk("[program2] : get SIGSEGV signal");
			printk("[program2] : child process has segemnt_fault");
			printk("[program2] : The return signal is 11");
		} else if (status == 4991) {
			printk("[program2] : get SIGSTOP signal");
			printk("[program2] : child process stopped");
			printk("[program2] : The return signal is 19");
		} else if (status == 15) {
			printk("[program2] : get SIGTERM signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is 15");
		} else if (status == 133) {
			printk("[program2] : get SIGTRAP signal");
			printk("[program2] : child process trapped");
			printk("[program2] : The return signal is 5");
		}
	}
	//printk("[program2] : The return signal is %d\n", status);

	put_pid(wo_pid);

	return;
}

//implement fork function
int my_fork(void *argc){
	
	//set default sigaction for current process
	int i;
	long pid = 0;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	printk("[program2] : module_init kthread start\n");
	pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);

	printk("[program2] : The child process has pid = %ld\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);

    my_wait(pid);
	
	return 0;
}

static int __init program2_init(void){

	printk("[program2] : module_init\n");
	
	/* create a kernel thread to run my_fork */
	task = kthread_create(&my_fork, NULL, "MyThread");

	if(!IS_ERR(task)){
		printk("[program2] : module_init create kthread start\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : module_exit./my\n");
}

module_init(program2_init);
module_exit(program2_exit);